#include "inferno/server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace inferno {
namespace {

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o.push_back('\\');
        o.push_back(c);
    }
    return o;
}

int parse_int_after(const std::string& body, const std::string& key, int fallback) {
    auto pos = body.find(key);
    if (pos == std::string::npos) return fallback;
    pos = body.find(':', pos + key.size());
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) ++pos;
    try {
        size_t used = 0;
        int v = std::stoi(body.substr(pos), &used);
        return v;
    } catch (...) {
        return fallback;
    }
}

std::uint64_t parse_u64_after(const std::string& body, const std::string& key, std::uint64_t fallback) {
    auto pos = body.find(key);
    if (pos == std::string::npos) return fallback;
    pos = body.find(':', pos + key.size());
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) ++pos;
    try {
        size_t used = 0;
        return std::stoull(body.substr(pos), &used);
    } catch (...) {
        return fallback;
    }
}

double parse_float_after(const std::string& body, const std::string& key, double fallback) {
    auto pos = body.find(key);
    if (pos == std::string::npos) return fallback;
    pos = body.find(':', pos + key.size());
    if (pos == std::string::npos) return fallback;
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) ++pos;
    try {
        size_t used = 0;
        return std::stod(body.substr(pos), &used);
    } catch (...) {
        return fallback;
    }
}

std::vector<int> parse_token_ids(const std::string& body) {
    auto pos = body.find("\"token_ids\"");
    if (pos == std::string::npos) throw std::runtime_error("missing token_ids");
    auto lb = body.find('[', pos);
    auto rb = body.find(']', lb == std::string::npos ? pos : lb);
    if (lb == std::string::npos || rb == std::string::npos) throw std::runtime_error("bad token_ids");
    std::vector<int> ids;
    std::string list = body.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
        size_t a = 0;
        while (a < item.size() && std::isspace(static_cast<unsigned char>(item[a]))) ++a;
        if (a == item.size()) continue;
        ids.push_back(std::stoi(item.substr(a)));
    }
    return ids;
}

void write_all(int fd, const std::string& data) {
    std::size_t off = 0;
    while (off < data.size()) {
        ssize_t n = ::send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n < 0) throw std::runtime_error("socket write failed");
        off += static_cast<std::size_t>(n);
    }
}

std::string read_request(int fd) {
    std::string data;
    char buf[4096];
    while (data.find("\r\n\r\n") == std::string::npos) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        data.append(buf, buf + n);
        if (data.size() > 2 * 1024 * 1024) throw std::runtime_error("request too large");
    }
    auto hdr = data.find("\r\n\r\n");
    if (hdr == std::string::npos) throw std::runtime_error("incomplete http header");
    std::string header = data.substr(0, hdr);
    std::size_t length = 0;
    auto cl = header.find("Content-Length:");
    if (cl == std::string::npos) cl = header.find("content-length:");
    if (cl != std::string::npos) {
        length = static_cast<std::size_t>(std::stoul(header.substr(cl + 15)));
    }
    std::string body = data.substr(hdr + 4);
    while (body.size() < length) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        body.append(buf, buf + n);
    }
    return header + "\r\n\r\n" + body.substr(0, length);
}

void respond(int fd, int code, const std::string& content_type, const std::string& body) {
    std::string status = code == 200 ? "OK" : "Bad Request";
    std::ostringstream o;
    o << "HTTP/1.1 " << code << ' ' << status << "\r\n"
      << "Content-Type: " << content_type << "\r\n"
      << "Content-Length: " << body.size() << "\r\n"
      << "Connection: close\r\n\r\n"
      << body;
    write_all(fd, o.str());
}

}  // namespace

std::string completions_response(const GenerationResult& result) {
    std::ostringstream o;
    o << "{\"id\":" << result.id << ",\"ok\":" << (result.ok ? "true" : "false")
      << ",\"generated\":" << result.generated << ",\"token_ids\":[";
    for (std::size_t i = 0; i < result.tokens.size(); ++i) {
        if (i) o << ',';
        o << result.tokens[i];
    }
    o << "]";
    if (!result.error.empty()) o << ",\"error\":\"" << json_escape(result.error) << "\"";
    o << "}";
    return o.str();
}

std::string metrics_text(const Metrics& m) {
    std::ostringstream o;
    o << "inferno_tokens_processed " << m.tokens_processed << "\n"
      << "inferno_preemptions " << m.preemptions << "\n"
      << "inferno_requests " << m.requests << "\n"
      << "inferno_steps " << m.steps << "\n"
      << "inferno_prefix_hits " << m.prefix_hits << "\n"
      << "inferno_prefix_misses " << m.prefix_misses << "\n"
      << "inferno_prefix_tokens_saved " << m.prefix_tokens_saved << "\n";
    return o.str();
}

GenerationResult run_completion_json(Engine& engine, const std::string& body) {
    Request req;
    req.prompt = parse_token_ids(body);
    req.max_new_tokens = parse_int_after(body, "\"max_tokens\"", 16);
    req.temperature = static_cast<float>(parse_float_after(body, "\"temperature\"", 0.0));
    req.top_k = parse_int_after(body, "\"top_k\"", 40);
    req.seed = parse_u64_after(body, "\"seed\"", 1);
    auto results = engine.generate({req});
    if (results.empty()) throw std::runtime_error("engine returned no result");
    return results[0];
}

HttpServer::HttpServer(Engine& engine, int port) : engine_(engine), port_(port) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::start() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket failed");
    int yes = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port_ < 0 ? 0 : port_));
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error("bind failed");
    }
    if (::listen(listen_fd_, 16) < 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error("listen failed");
    }
    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound), &len);
    port_ = ntohs(bound.sin_port);
    stop_ = false;
    thread_ = std::thread([this] { loop(); });
}

void HttpServer::stop() {
    stop_ = true;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
    }
    if (thread_.joinable()) thread_.join();
    {
        std::unique_lock<std::mutex> lock(idle_mu_);
        idle_cv_.wait(lock, [&] { return inflight_.load() == 0; });
    }
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
}

void HttpServer::loop() {
    while (!stop_) {
        pollfd pfd{listen_fd_, POLLIN, 0};
        int rc = ::poll(&pfd, 1, 200);
        if (rc <= 0) continue;
        int fd = ::accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) continue;
        inflight_.fetch_add(1);
        std::thread([this, fd] {
            handle(fd);
            if (inflight_.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lock(idle_mu_);
                idle_cv_.notify_all();
            }
        }).detach();
    }
}

void HttpServer::handle(int fd) {
    try {
        std::string raw = read_request(fd);
        auto line_end = raw.find("\r\n");
        std::string line = raw.substr(0, line_end);
        auto hdr_end = raw.find("\r\n\r\n");
        std::string body = hdr_end == std::string::npos ? std::string() : raw.substr(hdr_end + 4);
        if (line.find("GET /health") == 0) {
            respond(fd, 200, "application/json", "{\"status\":\"ok\"}");
        } else if (line.find("GET /metrics") == 0) {
            respond(fd, 200, "text/plain", metrics_text(engine_.metrics()));
        } else if (line.find("POST /v1/completions") == 0) {
            Request req;
            req.prompt = parse_token_ids(body);
            req.max_new_tokens = parse_int_after(body, "\"max_tokens\"", 16);
            req.temperature = static_cast<float>(parse_float_after(body, "\"temperature\"", 0.0));
            req.top_k = parse_int_after(body, "\"top_k\"", 40);
            req.seed = parse_u64_after(body, "\"seed\"", 1);
            GenerationResult result = engine_.submit(req).get();
            respond(fd, result.ok ? 200 : 400, "application/json", completions_response(result));
        } else {
            respond(fd, 400, "application/json", "{\"error\":\"unknown route\"}");
        }
    } catch (const std::exception& ex) {
        try {
            respond(fd, 400, "application/json",
                    std::string("{\"ok\":false,\"error\":\"") + json_escape(ex.what()) + "\"}");
        } catch (...) {
        }
    }
    ::close(fd);
}

}  // namespace inferno
