/*
 * rbe_package.sv
 * Gianna Paulin <pauling@iis.ee.ethz.ch>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2019-2021 ETH Zurich, University of Bologna
 * Copyright and related rights are licensed under the Solderpad Hardware
 * License, Version 0.51 (the "License"); you may not use this file except in
 * compliance with the License.  You may obtain a copy of the License at
 * http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
 * or agreed to in writing, software, hardware and materials distributed under
 * this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations under the License.
 *
 * Package of the Reconfigurable Binary Engine (RBE)
 *
 */


package rbe_package;

  // `define OBSERVE_TB

  // ========================================================================
  // PULP contents
  // ========================================================================

  parameter int NR_HWPE_REG   = 11;
  parameter int NR_HCI_REG    = 1;
  parameter int NR_UCODE_REG  = 12;

  // general PULP environment parameters including clusters etc
  // default number of cores
  parameter int NR_CORES = 8;

  // number of contexts
  parameter int NR_CONTEXT = 2;

  // default id width
  parameter int ID_WIDTH = 16;

  // number of registers
  parameter int NR_IO_REGS      = NR_HWPE_REG + NR_UCODE_REG; // 10 + 11 = 21
  parameter int NR_GENERIC_REGS = NR_HCI_REG;                 // 1

  // Maximum weight exponent offset (limits MAC bitwidths)
  // parameter int N2_MAX = 64;


  // ========================================================================
  // CTRL Registers
  // ========================================================================

  // ctrl counter bit-widths
  parameter int SPATIAL_CNT_SIZE   = 16;
  parameter int FILTER_CNT_SIZE    =  5;
  parameter int FEAT_CNT_SIZE      = 12;
  parameter int QUANT_CNT_SIZE     =  8;
  parameter int NB_ACC_CNT_SIZE    =  8;

  // uloop ctrl
  parameter int RBE_ULOOP_NB_LOOPS  =  5;
  parameter int RBE_ULOOP_LENGTH    = 28;
  parameter int RBE_ULOOP_NB_REG    =  9;
  parameter int RBE_ULOOP_NB_RO_REG = 11;
  parameter int RBE_ULOOP_REG_WIDTH = 32;
  parameter int RBE_ULOOP_CNT_WIDTH = 12;
  parameter int RBE_ULOOP_SHADOWED  =  0;

  // HWPE registers
  parameter int RBE_REG_X_ADDR         =  0;
  parameter int RBE_REG_W_ADDR         =  1;
  parameter int RBE_REG_NQ_ADDR        =  2;
  parameter int RBE_REG_Y_ADDR         =  3;
  parameter int RBE_REG_IMG_SIZE_INP   =  4;
  parameter int RBE_REG_IMG_SIZE_OUT   =  5;
  parameter int RBE_REG_CHANNELS       =  6;
  parameter int RBE_REG_FS             =  7;
  parameter int RBE_REG_QA_QW_QAO      =  8;
  parameter int RBE_REG_SHIFT          =  9;
  parameter int RBE_REG_HELPER         = 10;
  parameter int RBE_REG_UCODE_LOOPS0   = 11;
  parameter int RBE_REG_UCODE_LOOPS1   = 12;
  parameter int RBE_REG_UCODE_STATIC00 = 13;
  parameter int RBE_REG_UCODE_STATIC01 = 14;
  parameter int RBE_REG_UCODE_STATIC02 = 15;
  parameter int RBE_REG_UCODE_STATIC03 = 16;
  parameter int RBE_REG_UCODE_STATIC04 = 17;
  parameter int RBE_REG_UCODE_STATIC05 = 18;
  parameter int RBE_REG_UCODE_STATIC06 = 19;
  parameter int RBE_REG_UCODE_STATIC07 = 20;
  parameter int RBE_REG_UCODE_STATIC08 = 21;
  parameter int RBE_REG_UCODE_STATIC09 = 22;

  // GENERAL registers
  parameter int RBE_REG_HCI            = 0;  // HCI control

  // microcode offset indeces
  parameter int RBE_ULOOP_W_OFFS       =  0;
  parameter int RBE_ULOOP_X_OFFS       =  1;
  parameter int RBE_ULOOP_NQ_OFFS      =  2;
  parameter int RBE_ULOOP_Y_OFFS       =  3;

  parameter int RBE_ULOOP_MNEM_NIF            = 9  - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_NOF            = 10 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_PATCH_SIZE     = 11 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_PATCH_SIZE_W   = 12 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_PATCH_SIZE_H   = 13 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_PATCH_OUTPUT   = 14 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_PATCH_OUTPUT_W = 15 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_PATCH_OUTPUT_H = 16 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_STREAM_SIZE    = 17 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_NQ_BATCH       = 18 - RBE_ULOOP_NB_REG;
  parameter int RBE_ULOOP_MNEM_ZERO           = 19 - RBE_ULOOP_NB_REG;


  // ========================================================================
  // GENERAL parameters
  // ========================================================================
  // maximum quantization size
  parameter int MAX_QA            =  8;
  parameter int MAX_QW            =  8;
  parameter int MAX_QW_UNSIGNED   = MAX_QW - 1;

  parameter int MAX_SHIFT         = 16;

  // counter sizes
  parameter int ULOOP_CNT_WIDTH_1 = 12;

  // offset size
  parameter int OFFSET_SIZE       = 16;


  // ========================================================================
  // BANDWIDTH related types
  parameter int BITS_PER_TCDM_PORT = 32;
  parameter int NR_TCDM_PORTS      = 9;

  parameter int BANDWIDTH = BITS_PER_TCDM_PORT*NR_TCDM_PORTS; // 288bits (9 ports x 32 bits)


  // ========================================================================
  // FIFO Depths
  // usually never more than 2x256bits loaded
  parameter int FD_FEAT = 2;
  // always load all streaming data
  parameter int FD_WEIGHT = MAX_QW_UNSIGNED;
  // TODO check smallest possible value;
  parameter int FD_CONV = 2;


  // ========================================================================
  // BINCONV related types
  // Throughput parameter for a single BinConv module
  parameter int BINCONV_TP          = 32; // throughput parameter of BinConv

  // number of binary Sum-of-Products per BinConv block
  parameter int BINCONV_BLOCK_SIZE  =  4;

  // number of binary BinConv blocks per BinConv column
  parameter int BINCONV_COLUMN_SIZE =  9;

  // number of binary BinConv blocks per BinConv array
  parameter int BINCONV_NR_COLUMN   =  9;

  // number of Accumulators in the Accumulator Quantizor module
  parameter int NR_ACCUM            = BINCONV_TP;

  // feature buffer size
  parameter int BINCONV_FEAT_BUFFER_SIZE = 5;


  // ========================================================================
  // SCALE and COMBINATION module related types
  // SCALE module
  parameter int NR_SHIFTS          = 16;

  // COMBINATION module
  parameter int INTERNAL_ACCURACY  = 16;
  parameter int OUTPUT_ACCURACY    = 16;


  // ========================================================================
  // ACCUMULATOR module related types

  parameter int ACCUMULATOR_PARAMETER = BINCONV_TP; // #accumulators in the BinConv
  // parameter int unsigned MAX_ACCUMULATOR_PARAMETER  = 512;          // max limit of possible #accumulators in the BinConv

  parameter int ACCUMULATOR_SIZE        = 32;

  // number of bits used in vlen_cnt
  parameter int VLEN_CNT_SIZE           = 16;

  // (batch-)normalization parameters
  parameter int unsigned NORM_MULT_SIZE = 16;
  parameter int unsigned NORM_ADD_SIZE  = 32;


  // ========================================================================
  // FEAT_BUFFER related types
  // ========================================================================
  typedef struct packed {
    logic                     goto_load;
    logic                     goto_extract;
    logic [VLEN_CNT_SIZE-1:0] i_vlen;       // virtual buffer length
    logic [VLEN_CNT_SIZE-1:0] o_vlen;
  } ctrl_feat_buf_t;

  typedef enum {
    FR_IDLE, FR_LOAD, FR_EXTRACT
  } state_feat_buf_t;

  typedef struct packed {
    state_feat_buf_t state;
  } flags_feat_buf_t;


  // ========================================================================
  // SIGN_BUFFER related types
  // ========================================================================
  typedef struct packed {
    logic                     goto_load;
    logic                     goto_extract;
    logic [VLEN_CNT_SIZE-1:0] i_vlen;       // virtual buffer length
    logic [VLEN_CNT_SIZE-1:0] o_vlen;
  } ctrl_sign_buf_t;

  typedef enum {
    SR_IDLE, SR_LOAD, SR_EXTRACT
  } state_sign_buf_t;

  typedef struct packed {
    state_sign_buf_t state;
  } flags_sign_buf_t;


  // ========================================================================
  // SOP related types
  // ========================================================================
  typedef struct packed {
    logic                  operation_sel; // 1:xnor, 0:and
    logic [BINCONV_TP-1:0] inactive_mask;
    logic                  clear;
  } ctrl_sop_t;


  // ========================================================================
  // Accumulator Quantizor related types
  // ========================================================================

  typedef struct packed {
    logic                            start;
    logic                            relu;
    logic [4:0]                      right_shift;
    logic [$clog2(QUANT_CNT_SIZE):0] qa_out;
  } ctrl_normquant_t;

  typedef struct packed {
    logic ready;
  } flags_normquant_t;

  typedef struct packed {
    logic [        BINCONV_TP/8-1:0] mask;                    // TODO: only strb connected
    logic [       VLEN_CNT_SIZE-1:0] single_accumulation_len; // nr of read out values
    logic [       VLEN_CNT_SIZE-1:0] full_accumulation_len;   // nr of accumulations
    logic                            clear;
    logic [$clog2(QUANT_CNT_SIZE):0] qa_out;       // output quantization
    logic [$clog2(QUANT_CNT_SIZE):0] n_accum;
    logic                            offset_en;    // offset computation enables
    logic                            offset_state; // 0: shift = offset_shift 1: shift = identity_shift
    logic                            goto_norm;
    logic                            goto_accum;
    ctrl_normquant_t                 ctrl_normquant;
  } ctrl_aq_t;

  typedef enum {
    AQ_IDLE, AQ_ACCUMULATE, AQ_ACCUM_DONE, AQ_NORMQUANT, AQ_NORMQUANT_DONE,
    AQ_STREAMOUT, AQ_STREAMOUT_DONE
  } state_aq_t;

  typedef struct packed {
    state_aq_t  state;
    logic [1:0] norm_cnt;
    logic [2:0] bit_cnt;
  } flags_aq_t;

  // ========================================================================
  // SCALE related types
  // ========================================================================

  typedef struct packed {
    logic [$clog2(MAX_SHIFT):0] shift_sel;
  } ctrl_scale_t;

  typedef struct packed {
    logic [$clog2(MAX_SHIFT):0] shift_sel;
  } flags_scale_t;

  // ========================================================================
  // BINCONV_BLOCK related types
  // ========================================================================

  typedef struct packed {
    ctrl_sop_t                       ctrl_sop;
    logic [$clog2(QUANT_CNT_SIZE):0] qw;
    logic                            offset_en;       // offset computation enables
    logic                            offset_state;    // 0: shift = offset_shift 1: shift = identity_shift
    logic [             $clog2(7):0] fs;              // filter size
    logic [ BINCONV_COLUMN_SIZE-1:0] row_onehot_en;   // select working blocks
    logic [                     1:0] qa_tile_sel;
  } ctrl_binconv_block_t;

  typedef struct packed {
    flags_scale_t [BINCONV_BLOCK_SIZE-1:0] flags_scale;
  } flags_binconv_block_t;

  // ========================================================================
  // BINCONV_COLUMN related types
  // ========================================================================

  typedef struct packed {
    ctrl_binconv_block_t ctrl_block;
  } ctrl_binconv_column_t;

  typedef struct packed {
    flags_binconv_block_t [BINCONV_COLUMN_SIZE-1:0] flags_block;
  } flags_binconv_column_t;

  // ========================================================================
  // BINCONV_ARRAY related types
  // ========================================================================

  typedef struct packed {
    ctrl_binconv_column_t ctrl_column;
    logic [$clog2(7):0]   fs;
  } ctrl_binconv_array_t;

  typedef struct packed {
    flags_binconv_column_t [BINCONV_NR_COLUMN-1:0] flags_column;
  } flags_binconv_array_t;


  // ========================================================================
  // ENGINE_CORE related types
  // ========================================================================

  parameter int BUFFER_PATCH_SIZE_MAX = 5;
  parameter int BUFFER_PATCH_SIZE_MIN = 3;

  parameter int NR_A_STREAMS_MAX = BUFFER_PATCH_SIZE_MAX*BINCONV_BLOCK_SIZE;
  parameter int NR_A_STREAMS_MIN = BUFFER_PATCH_SIZE_MIN*BINCONV_BLOCK_SIZE;

  parameter int X_OFFSET_MAX = NR_A_STREAMS_MAX;
  parameter int X_OFFSET_MIN = NR_A_STREAMS_MIN;

  parameter int NR_ACTIVATIONS = BUFFER_PATCH_SIZE_MAX*BUFFER_PATCH_SIZE_MAX*BINCONV_BLOCK_SIZE;

  typedef struct packed {
    ctrl_feat_buf_t      ctrl_activation_reg;
    ctrl_binconv_array_t ctrl_array;
    ctrl_aq_t            ctrl_aq;
  } ctrl_engine_core_t;

  typedef struct packed {
    flags_feat_buf_t [NR_ACTIVATIONS-1:0]    flags_activation_reg;
    flags_aq_t       [BINCONV_NR_COLUMN-1:0] flags_aq;
    flags_binconv_array_t                    flags_binconv_array;
  } flags_engine_core_t;

  // ========================================================================
  // ENGINE related types
  // ========================================================================

  typedef struct packed {
    ctrl_engine_core_t               ctrl_engine_core;
    logic [$clog2(QUANT_CNT_SIZE):0] qa_out;
    logic [$clog2(7):0]              fs;
  } ctrl_engine_t;

  typedef struct packed {
    flags_engine_core_t                       flags_engine_core;
    logic                                     feat_fence_arrived;
    logic                                     valid_weight_hs;
    logic [                              1:0] sel_c_out_pixel;
    logic [   $clog2(BINCONV_BLOCK_SIZE)-1:0] sel_bitidx_demux;
    logic [$clog2(BUFFER_PATCH_SIZE_MAX)-1:0] sel_pixel_demux;
  } flags_engine_t;

  // ========================================================================
  // URISCY CTRL related types
  // ========================================================================

  typedef struct packed {
    logic start;
  } ctrl_ctrlmult_t;

  typedef struct packed {
    logic valid;
  } flags_ctrlmult_t;


  // ========================================================================
  // STREAMER related types
  // ========================================================================

  typedef enum logic [1:0] { LD_FEAT_SEL, LD_WEIGHT_SEL, LD_NORM_SEL } ld_which_mux_sel_e;

  typedef struct packed {
    ld_which_mux_sel_e               ld_which_mux_sel;
    logic                            ld_st_mux_sel;
    hci_package::hci_streamer_ctrl_t feat_source_ctrl;
    hci_package::hci_streamer_ctrl_t weight_source_ctrl;
    hci_package::hci_streamer_ctrl_t norm_source_ctrl;
    hci_package::hci_streamer_ctrl_t conv_sink_ctrl;
  } ctrl_streamer_t;

  typedef struct packed {
    hci_package::hci_streamer_flags_t feat_source_flags;
    hci_package::hci_streamer_flags_t weight_source_flags;
    hci_package::hci_streamer_flags_t norm_source_flags;
    hci_package::hci_streamer_flags_t conv_sink_flags;
    logic tcdm_fifo_empty;
  } flags_streamer_t;


  // ========================================================================
  // CTRL FSM related types
  // ========================================================================

  typedef struct packed {
    logic [                $clog2(QUANT_CNT_SIZE):0] qw;
    logic [                $clog2(QUANT_CNT_SIZE):0] qa;
    logic [                $clog2(QUANT_CNT_SIZE):0] qa_out;
    logic [                             $clog2(7):0] fs;
    logic [                       FEAT_CNT_SIZE-1:0] kin;
    logic [                       FEAT_CNT_SIZE-1:0] kout;
    logic [                 RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_K_in;
    logic [                 RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_K_out;
    logic [                 RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_qa;
    logic [                 RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_qa_out;
    logic [               RBE_ULOOP_CNT_WIDTH+2-1:0] n_tiles_K_in_qa;
    logic [               RBE_ULOOP_CNT_WIDTH+2-1:0] n_tiles_K_out_qa_out;
    logic [SPATIAL_CNT_SIZE+RBE_ULOOP_CNT_WIDTH+1:0] static_w_tiles_K_in_qa;
    logic [SPATIAL_CNT_SIZE+RBE_ULOOP_CNT_WIDTH+1:0] static_w_tiles_K_out_qa_out;
    logic [                    SPATIAL_CNT_SIZE-1:0] iw;
    logic [                    SPATIAL_CNT_SIZE-1:0] ih;
    logic [                    SPATIAL_CNT_SIZE-1:0] ow;
    logic [                                  16-1:0] shift;
    logic                                            relu;
  } ctrl_ctrlfsm_t;

  typedef enum logic [3:0] {
    RBECTRL_IDLE,
    RBECTRL_WAITMULT,
    RBECTRL_START,
    RBECTRL_LOAD_FEAT,
    RBECTRL_COMPUTE,
    RBECTRL_NORMQUANT,
    RBECTRL_STREAMOUT,
    RBECTRL_STREAMOUT_ENDING,
    RBECTRL_UPDATEIDX, RBECTRL_WAIT,
    RBECTRL_TERMINATE
  } state_rbectrl_e;

endpackage
