################################################################################
#####                                                                          #
#####     SETTINGS                                                             #
#####                                                                          #
################################################################################
## Languages                                                                   #
################################################################################
PYTHON = python3.6


################################################################################
## Design Parameters                                                           #
################################################################################
TP              = 32

BLOCK_SIZE      = 4
COLUMN_SIZE     = 9
N_COLUMNS       = 9

B_ACCUM_WIDTH   ?= 32
C_ACCUM_WIDTH   ?= 32

H_OUT           ?= 3
W_OUT           ?= 3
K_OUT           ?= 32
K_IN            ?= 32
FS              ?= 3

QA              ?= 4
QW              ?= 4
QAO             ?= 4

VERIFY          ?= 1
RELU            ?= 1


################################################################################
## Paths and relations                                                         #
################################################################################
# golden model
GOLDEN_MODEL_PATH = ./model/
GOLDEN_MODEL      = run_golden_model.py


################################################################################
#####                                                                          #
#####     TARGETS                                                              #
#####                                                                          #
################################################################################
.PHONY: synth clean% update all ucode


################################################################################
## Golden Model and Stimuli Targets                                            #
################################################################################

# create the directory to store stimuli and golden vectors
dirs:
	mkdir -p $(GOLDEN_MODEL_PATH)golden_bittrue
	mkdir -p $(GOLDEN_MODEL_PATH)golden_simple
	mkdir -p $(GOLDEN_MODEL_PATH)golden_cmp

stimuli: dirs
	cd $(GOLDEN_MODEL_PATH) && $(PYTHON) ./$(GOLDEN_MODEL) \
			--tp      $(TP) \
			--bs      $(BLOCK_SIZE) \
			--colsize $(COLUMN_SIZE) \
			--nrcol   $(N_COLUMNS) \
			--kin     $(K_IN) \
			--kout    $(K_OUT) \
			--hout    $(H_OUT) \
			--wout    $(W_OUT) \
			--fs      $(FS) \
			--b_accum_width $(B_ACCUM_WIDTH) \
			--c_accum_width $(C_ACCUM_WIDTH) \
			--qa      $(QA) \
			--qao     $(QAO) \
			--qw      $(QW) \
			--v       $(VERIFY) \
            --relu    $(RELU)

ucode:
	cd ./ucode && $(PYTHON) uloop_compile.py


################################################################################
## Cleanup Targets                                                             #
################################################################################

clean-stimuli:
	rm -rf $(GOLDEN_MODEL_PATH)/golden_bittrue
	rm -rf $(GOLDEN_MODEL_PATH)/golden_simple
	rm -rf $(GOLDEN_MODEL_PATH)/golden_cmp

clean: clean-stimuli


################################################################################
## Help targets
################################################################################

help :

	@echo "USAGE      : make [options] <target(s)> (<parameter(s)>)"
	@echo
	@echo "TARGETS    : help    - Show the help."
	@echo
	@echo "             all     - make clean stimuli"
	@echo
	@echo "             clean   - Clean the stimuli directory."
	@echo
	@echo "             dirs    - Create directory for stimuli generation"
	@echo "             stimuli - Generate the stimuli."
	@echo "             ucode   - Compile the ucode used for the rbe"
	@echo
	@echo "PARAMETERS :"
	@echo "             TP          = 32 - fixed by HW, working on 32 parallel activation channels"
	@echo "             BLOCK_SIZE  = 4  - fixed by HW, each block has 4 parallel BinConv engines"
	@echo "             COLUMN_SIZE = 9  - fixed by HW, each column includes 9 Blocks"
	@echo "             N_COLUMNS   = 9  - fixed by HW, RBE engine includes 9 Columns of Blocks"
	@echo
	@echo "             B_ACCUM_WIDTH ?= 32  - fixed by HW, configurable for golden model: Bitwidth of BinConv accumulators"
	@echo "             C_ACCUM_WIDTH ?= 32  - fixed by HW, configurable for golden model: Bitwidth of Column accumulators"
	@echo
	@echo "             VERIFY ?= 0 - golden model, verify bittrue model against simple convolution model"
	@echo
	@echo "             K_IN  ?= 32 - network to be run, # input channels, can be multiple of 32"
	@echo "             K_OUT ?= 32 - network to be run, # output channels, can be multiple of 32"
	@echo "             H_OUT ?= 3  - network to be run, output image height, can be multiple of 3"
	@echo "             W_OUT ?= 3  - network to be run, output image width, can be multiple of 3"
	@echo "             FS    ?= 3  - network to be run, filter size, can be 3 (3x3), or 1 (1x1)"
	@echo "             RELU  ?= 1  - network to be run, enable or disable ReLU activation function"
	@echo
	@echo "             QA  ?= 4 - network to be run, #bits for input activations, can be between and including 2 and 8"
	@echo "             QW  ?= 4 - network to be run, #bits for weights, can be between and including 2 and 8"
	@echo "             QAO ?= 4 - network to be run, #bits for output activations, can be between and including 2 and 8"
