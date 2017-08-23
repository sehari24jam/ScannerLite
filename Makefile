.DEFAULT_GOAL = all
.PHONY: all clean

BIN = scannerLite
C++ := clang
C++FLAGS +=  -lopencv_core -lopencv_imgproc -lopencv_highgui -lboost_program_options -lstdc++

clean:
	@rm -f $(BIN)

all: $(BIN)

% : %.cpp
	@$(C++) $< -o $@ $(C++FLAGS)
