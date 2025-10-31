CXX      := g++

PKGLIBS = $(shell pkg-config --libs libcaer opencv4)
PKGFLAGS = $(shell pkg-config --cflags libcaer opencv4)

CXXFLAGS := -g -Wall -Wextra -O2 -D_DEFAULT_SOURCE=1 $(PKGFLAGS)
LDLIBS   := $(PKGLIBS)

SRCDIR   := ../src
INCDIR   := ../include
BUILDDIR := ../build

SRC_MAIN := main.cpp
SRC_LIB  := $(wildcard $(SRCDIR)/*.cpp)

OBJS := $(BUILDDIR)/main.o $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRC_LIB))

TARGET := main

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $^ -o $@ $(LDLIBS)

$(BUILDDIR)/main.o: main.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -c $< -o $@

$(BUILDDIR):
	mkdir -p $@
clean:
	rm -rf $(BUILDDIR) $(TARGET)
