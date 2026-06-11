# csrc/categorizer/Makefile — builds libcategorizer (KMeans discretizer)
#
# Usage:
#   make          → build shared library
#   make install  → build + copy to lbtree/_preprocessing/_backend/
#   make clean    → remove build artefacts

CC      = gcc
CFLAGS  = -fPIC -shared -O2 -Wall
LDFLAGS = -lm
SOURCES = categorizer_core.c kmeans.c kmeans_init.c kmeans_metrics.c
INSTALL_DIR = ../../lbtree/_preprocessing/_backend

# ---- Platform detection ----
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB = libcategorizer.so
endif
ifeq ($(UNAME_S),Darwin)
    LIB = libcategorizer.dylib
endif
ifndef LIB
    LIB = libcategorizer.so
endif

.PHONY: all install clean

all: $(LIB)

$(LIB): $(SOURCES) categorizer.h kmeans.h
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS)
	@echo "✓  Built $(LIB)"

install: $(LIB)
	@mkdir -p $(INSTALL_DIR)
	@cp $(LIB) $(INSTALL_DIR)/
	@echo "✓  Installed $(LIB) → $(INSTALL_DIR)"

clean:
	@rm -f $(LIB)
	@rm -f $(INSTALL_DIR)/libcategorizer.*
	@echo "✓  Cleaned"
