# csrc/pitree/Makefile — builds libpitree (PredictabilityTree + AdaBoost backend)
#
# Usage:
#   make          → build shared library
#   make install  → build + copy to sltree/_backend/
#   make clean    → remove build artefacts

CC      = cc
CFLAGS  = -fPIC -shared -O2 -Wall
LDFLAGS = -lm
SRC     = pitree_core.c
INSTALL_DIR = ../../sltree/_backend

# ---- Platform detection ----
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB = libpitree.so
endif
ifeq ($(UNAME_S),Darwin)
    LIB = libpitree.dylib
endif
ifndef LIB
    # fallback (Windows cross-compile or unknown)
    LIB = libpitree.so
endif

.PHONY: all install clean

all: $(LIB)

$(LIB): $(SRC) pitree_core.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "✓  Built $(LIB)"

install: $(LIB)
	@mkdir -p $(INSTALL_DIR)
	@cp $(LIB) $(INSTALL_DIR)/
	@echo "✓  Installed $(LIB) → $(INSTALL_DIR)"

clean:
	@rm -f $(LIB)
	@rm -f $(INSTALL_DIR)/libpitree.*
	@echo "✓  Cleaned"
