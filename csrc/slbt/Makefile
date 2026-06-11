# csrc/slbt/Makefile — builds libslbt (SLBT backend)
#
# Usage:
#   make          → build shared library
#   make install  → build + copy to lbtree/_backend/
#   make clean    → remove build artefacts

CC      = gcc
CFLAGS  = -fPIC -shared -O3 -Wall -Wextra
LDFLAGS = -lm
SOURCES = slbt_core.c slbt_als.c linalg.c
INSTALL_DIR = ../../lbtree/_backend

# ---- Platform detection ----
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB = libslbt.so
endif
ifeq ($(UNAME_S),Darwin)
    LIB = libslbt.dylib
endif
ifndef LIB
    LIB = libslbt.so
endif

.PHONY: all install clean

all: $(LIB)

$(LIB): $(SOURCES) slbt.h slbt_als.h linalg.h
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS)
	@echo "✓  Built $(LIB)"

install: $(LIB)
	@mkdir -p $(INSTALL_DIR)
	@cp $(LIB) $(INSTALL_DIR)/
	@echo "✓  Installed $(LIB) → $(INSTALL_DIR)"

clean:
	@rm -f $(LIB)
	@rm -f $(INSTALL_DIR)/libslbt.*
	@echo "✓  Cleaned"
