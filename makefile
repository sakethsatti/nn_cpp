# Compiler
CXX = g++

# Source files
SRCS = src/titanic/train_titanic.cpp src/titanic/load_titanic.cpp src/nn.cpp src/layer.cpp

# Target
TARGET = nn_run

# Build rule
$(TARGET): $(SRCS)
	$(CXX) $(SRCS) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET)
