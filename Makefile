COMPILER ?= g++
BUILD ?= build

.PHONY: all test bench demo clean

all:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$(COMPILER)
	cmake --build $(BUILD) -j

test: all
	./$(BUILD)/inferno_tests

bench: all
	./$(BUILD)/inferno bench

demo: all
	./$(BUILD)/inferno demo

clean:
	rm -rf $(BUILD)
