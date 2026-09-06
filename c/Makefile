# VITS TTS (mms-tts-por) — Pure C implementation
#
# Build:
#   make              # standard build
#   make DUMP=1       # build with --dump-dir validation support
#   make run          # build and run with default sample
#   make validate     # build ref + C dumps, then compare
#   make clean

CC       ?= gcc
CFLAGS   ?= -O2 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  ?= -lm

SRC_DIR  = src
VAL_DIR  = validate
BIN      = mms-tts

ifneq ($(OMP),)
  CFLAGS  += -fopenmp
  LDFLAGS += -fopenmp
endif

SRCS = $(SRC_DIR)/main.c \
       $(SRC_DIR)/ops.c \
       $(SRC_DIR)/tokenizer.c \
       $(SRC_DIR)/encoder.c \
       $(SRC_DIR)/duration.c \
       $(SRC_DIR)/flow.c \
       $(SRC_DIR)/hifigan.c \
       $(SRC_DIR)/vtsm.c \
       $(SRC_DIR)/model.c \
       $(SRC_DIR)/wav.c \
       $(SRC_DIR)/npy_reader.c

ifneq ($(DUMP),)
  CFLAGS += -DENABLE_DUMP
  SRCS   += $(VAL_DIR)/dump.c
endif

OBJS = $(SRCS:.c=.o)

.PHONY: all run validate clean

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c $(SRC_DIR)/vits.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(SRC_DIR)/vtsm.o: model_weights.h

run: $(BIN)
	./$(BIN)

# Validation: generate Python reference, C dump, and compare
validate:
	$(MAKE) clean
	$(MAKE) DUMP=1 $(BIN)
	python3 $(VAL_DIR)/stages_ref.py --text "Olá, mundo!" --seed 42 --out-dir ref_out
	mkdir -p c_out
	./$(BIN) --text "Olá, mundo!" --seed 42 --inject-dir ref_out --dump-dir c_out
	python3 $(VAL_DIR)/compare.py --ref-dir ref_out --c-dir c_out

clean:
	rm -f $(OBJS) $(BIN)
