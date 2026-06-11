# csrc/lbtree/Makefile — builds liblbtree (SCTree + AdaBoost backend)
#
# Usage:
#   make          → build shared library
#   make install  → build + copy to lbtree/_backend/
#   make clean    → remove build artefacts

CC      = cc
CFLAGS  = -fPIC -shared -O2 -Wall
LDFLAGS = -lm
SRC     = lbtree_core.c
INSTALL_DIR = ../../lbtree/_backend

# ---- Platform detection ----
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB = liblbtree.so
endif
ifeq ($(UNAME_S),Darwin)
    LIB = liblbtree.dylib
endif
ifndef LIB
    # fallback (Windows cross-compile or unknown)
    LIB = liblbtree.so
endif

.PHONY: all install clean

all: $(LIB)

$(LIB): $(SRC) lbtree_core.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓  Built $(LIB)"

install: $(LIB)
	@mkdir -p $(INSTALL_DIR)
	@cp $(LIB) $(INSTALL_DIR)/
	@echo "✓  Installed $(LIB) → $(INSTALL_DIR)"

clean:
	@rm -f $(LIB)
	@rm -f $(INSTALL_DIR)/liblbtree.*
	@echo "✓  Cleaned"
