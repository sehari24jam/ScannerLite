.DEFAULT_GOAL = all
.PHONY: all clean

BIN = scannerLite
C++ := clang
C++FLAGS += -lopencv_core -lopencv_imgproc -lopencv_highgui
C++FLAGS += -lboost_program_options
C++FLAGS += -lboost_system -lboost_filesystem
C++FLAGS += -lstdc++

clean:
	@rm -f $(BIN)

all: $(BIN)

% : %.cpp
	@$(C++) $< -o $@ $(C++FLAGS)
