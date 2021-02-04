/*
 * rbe_ctrl.sv
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
 * TODO Description
 *
 */


module rbe_ctrl
  import rbe_package::*;
#(
  parameter int unsigned N_CORES        = rbe_package::NR_CORES,
  parameter int unsigned N_CONTEXT      = rbe_package::NR_CONTEXT,
  parameter int unsigned N_IO_REGS      = rbe_package::NR_IO_REGS,
  parameter int unsigned N_GENERIC_REGS = rbe_package::NR_GENERIC_REGS,
  parameter int unsigned ID_WIDTH       = rbe_package::ID_WIDTH,
  parameter int unsigned TP             = rbe_package::BINCONV_TP
) (
  // global signals
  input  logic                                                     clk_i,
  input  logic                                                     rst_ni,
  input  logic                                                     test_mode_i,
  output logic                                                     clear_o,
  output hci_package::hci_interconnect_ctrl_t                      ctrl_hci_o,
  // events
  output logic [N_CORES-1:0][hwpe_ctrl_package::REGFILE_N_EVT-1:0] evt_o,
  // ctrl & flags
  output ctrl_streamer_t                                           ctrl_streamer_o,
  input  flags_streamer_t                                          flags_streamer_i,
  output ctrl_engine_t                                             ctrl_engine_o,
  input  flags_engine_t                                            flags_engine_i,
  // periph slave port
  hwpe_ctrl_intf_periph.slave                                      periph
);

  /////////////
  // SIGNALS //
  /////////////

  // ctrl and flag signals
  hwpe_ctrl_package::ctrl_slave_t   slave_ctrl;
  hwpe_ctrl_package::flags_slave_t  slave_flags;

  hwpe_ctrl_package::ctrl_uloop_t   uloop_ctrl;
  hwpe_ctrl_package::flags_uloop_t  uloop_flags;

  hwpe_ctrl_package::uloop_code_t   uloop_code;
  hwpe_ctrl_package::ctrl_regfile_t reg_file;

  ctrl_ctrlmult_t  ctrl_ctrlmult;
  flags_ctrlmult_t flags_ctrlmult;

  ctrl_ctrlfsm_t   fsm_ctrl;

  // static register signals
  logic [      FILTER_CNT_SIZE-1:0] static_reg_fs0;
  logic [RBE_ULOOP_REG_WIDTH/2-1:0] static_reg_fs0_fs0_qw;

  logic [  FEAT_CNT_SIZE-1:0] static_reg_kin;
  logic [  FEAT_CNT_SIZE-1:0] static_reg_kout;

  logic [SPATIAL_CNT_SIZE-1:0] static_reg_w;
  logic [SPATIAL_CNT_SIZE-1:0] static_reg_h;
  logic [SPATIAL_CNT_SIZE-1:0] static_reg_ow;
  logic [SPATIAL_CNT_SIZE-1:0] static_reg_oh;

  logic [  QUANT_CNT_SIZE-1:0] static_reg_qw;
  logic [  QUANT_CNT_SIZE-1:0] static_reg_qa_m1;
  logic [  QUANT_CNT_SIZE-1:0] static_reg_qa_out;

  logic [RBE_ULOOP_CNT_WIDTH-1:0] static_reg_n_xpatches_w;
  logic [RBE_ULOOP_CNT_WIDTH-1:0] static_reg_n_xpatches_h;

  logic [16-1:0] static_reg_shift;
  logic          static_reg_relu;

  logic [RBE_ULOOP_CNT_WIDTH+2-1:0] static_tiles_K_in_qa;
  logic [RBE_ULOOP_CNT_WIDTH+2-1:0] static_tiles_K_out_qa_out;

  logic [SPATIAL_CNT_SIZE+RBE_ULOOP_CNT_WIDTH+1:0]  static_w_tiles_K_in_qa;
  logic                                             static_w_tiles_K_in_qa_valid;

  logic [SPATIAL_CNT_SIZE+RBE_ULOOP_CNT_WIDTH+1:0] static_w_tiles_K_out_qa_out;
  logic                                            static_w_tiles_K_out_qa_out_valid;

  logic [QUANT_CNT_SIZE+2*$clog2(TP)+1:0] static_fs0_qw;
  logic                                   static_fs0_qw_valid;

  logic [QUANT_CNT_SIZE+$clog2(TP)+RBE_ULOOP_REG_WIDTH/2:0] static_fs0_fs0_qw_TP2;
  logic                                                     static_fs0_fs0_qw_TP2_valid;


  /////////////////////
  // HWPE CTRL SLAVE //
  /////////////////////

  hwpe_ctrl_slave #(
    .N_CORES        ( N_CORES        ),
    .N_CONTEXT      ( N_CONTEXT      ),
    .N_IO_REGS      ( N_IO_REGS      ),
    .N_GENERIC_REGS ( N_GENERIC_REGS ),
    .ID_WIDTH       ( ID_WIDTH       )
  ) i_slave (
    .clk_i    ( clk_i       ),
    .rst_ni   ( rst_ni      ),
    .clear_o  ( clear_o     ),
    .cfg      ( periph      ),
    .ctrl_i   ( slave_ctrl  ),
    .flags_o  ( slave_flags ),
    .reg_file ( reg_file    )
  );
  assign evt_o = slave_flags.evt;

  /////////////////////////////////////////////
  // STATIC REGISTERS (RBE CONFIG REGISTERS) //
  /////////////////////////////////////////////
  // input width
  // input height
  assign static_reg_w  = reg_file.hwpe_params[RBE_REG_IMG_SIZE_INP][31:16] + 1; // 1-512px
  assign static_reg_h  = reg_file.hwpe_params[RBE_REG_IMG_SIZE_INP][15: 0] + 1; // 1-512px

  // output width
  // output height
  assign static_reg_ow = reg_file.hwpe_params[RBE_REG_IMG_SIZE_OUT][31:16] + 1; // 1-512px
  assign static_reg_oh = reg_file.hwpe_params[RBE_REG_IMG_SIZE_OUT][15: 0] + 1; // 1-512px

  // number of input feature maps
  // number of output feature maps
  assign static_reg_kin  = reg_file.hwpe_params[RBE_REG_CHANNELS][31:16] + 1; // 1-4096 input features
  assign static_reg_kout = reg_file.hwpe_params[RBE_REG_CHANNELS][15: 0] + 1; // 1-4096 output features

  // filter size
  assign static_reg_fs0        = reg_file.hwpe_params[RBE_REG_FS][23:16] + 1; // 1 / 3 supported
  assign static_reg_fs0_fs0_qw = reg_file.hwpe_params[RBE_REG_FS][15:0]  + 1; // 1 / 9 supported

  // quantization weights
  // quantization activations
  assign static_reg_qa_out = reg_file.hwpe_params[RBE_REG_QA_QW_QAO][24:16] + 1; // 1-8 bit supported
  assign static_reg_qw     = reg_file.hwpe_params[RBE_REG_QA_QW_QAO][15: 8] + 1; // 1-8 bit supported
  assign static_reg_qa_m1  = reg_file.hwpe_params[RBE_REG_QA_QW_QAO][ 7: 0];     // 1-8 bit supported

  // normquant parameters
  assign static_reg_relu  = reg_file.hwpe_params[RBE_REG_SHIFT][31];
  assign static_reg_shift = reg_file.hwpe_params[RBE_REG_SHIFT][15:0] + 1;

  // number of tiles in height and width
  assign static_reg_n_xpatches_w = reg_file.hwpe_params[       RBE_REG_FS][31:24] + 1;
  assign static_reg_n_xpatches_h = reg_file.hwpe_params[RBE_REG_QA_QW_QAO][31:24] + 1;

  // n_tiles_qa * n_tiles_k_in
  assign static_tiles_K_in_qa      = reg_file.hwpe_params[RBE_REG_HELPER][31:16]+1;
  assign static_tiles_K_out_qa_out = reg_file.hwpe_params[RBE_REG_HELPER][15: 0]+1;


  /////////////////////////////////////////////////
  // STATIC REGISTERS (GENERAL CONFIG REGISTERS) //
  /////////////////////////////////////////////////
  // HCI Ctrl
  assign ctrl_hci_o.low_prio_max_stall = reg_file.generic_params[RBE_REG_HCI][31:24]; // 8 bits
  assign ctrl_hci_o.arb_policy         = reg_file.generic_params[RBE_REG_HCI][18:17]; // 2 bits
  assign ctrl_hci_o.hwpe_prio          = reg_file.generic_params[RBE_REG_HCI][16];    // 1 bits


  ////////////////////////////////////////
  // COMPUTE INTERMEDIATE CONFIGURATION //
  ////////////////////////////////////////
  // signals
  logic [RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_qa;
  logic [RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_qa_out;
  logic [RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_K_in;
  logic [RBE_ULOOP_CNT_WIDTH-1:0] n_tiles_K_out;
  logic [RBE_ULOOP_REG_WIDTH-1:0] patch_output_h;
  logic [RBE_ULOOP_REG_WIDTH-1:0] stream_size;

  // compute tiles and reused values from configuration registers
  assign n_tiles_qa     = (static_reg_qa_m1 >> 2) + 1;
  assign n_tiles_qa_out = ((static_reg_qa_out-1) >> 2) + 1;
  assign n_tiles_K_in   = ((static_reg_kin-1) >> 5) + 1;
  assign n_tiles_K_out  = ((static_reg_kout-1) >> 5) + 1;

  assign patch_output_h = (static_w_tiles_K_out_qa_out*TP*(BUFFER_PATCH_SIZE_MIN)<<2);
  assign stream_size    = (static_reg_fs0 == 1) ? static_fs0_qw : static_fs0_fs0_qw_TP2;


  ////////////////////////////////
  // QUANTIZATION BASED CONTROL //
  ////////////////////////////////

  assign fsm_ctrl.qw                     = static_reg_qw;
  assign fsm_ctrl.qa                     = static_reg_qa_m1+1;
  assign fsm_ctrl.qa_out                 = static_reg_qa_out;
  assign fsm_ctrl.fs                     = static_reg_fs0;
  assign fsm_ctrl.kout                   = static_reg_kout;
  assign fsm_ctrl.kin                    = static_reg_kin;
  assign fsm_ctrl.n_tiles_K_out          = n_tiles_K_out;
  assign fsm_ctrl.n_tiles_K_in           = n_tiles_K_in;
  assign fsm_ctrl.n_tiles_qa             = n_tiles_qa;
  assign fsm_ctrl.n_tiles_qa_out         = n_tiles_qa_out;
  assign fsm_ctrl.n_tiles_K_in_qa        = static_tiles_K_in_qa;
  assign fsm_ctrl.n_tiles_K_out_qa_out   = static_tiles_K_out_qa_out;
  assign fsm_ctrl.static_w_tiles_K_out_qa_out = static_w_tiles_K_out_qa_out;
  assign fsm_ctrl.iw                     = static_reg_w;
  assign fsm_ctrl.ow                     = static_reg_ow;
  assign fsm_ctrl.ih                     = static_reg_h;
  assign fsm_ctrl.shift                  = static_reg_shift;
  assign fsm_ctrl.relu                   = static_reg_relu;
  assign fsm_ctrl.static_w_tiles_K_in_qa = static_w_tiles_K_in_qa;


  //////////////
  // CTRL FSM //
  //////////////

  rbe_ctrl_fsm #(
    .TP ( TP )
  ) i_ctrl_fsm (
    .clk_i            ( clk_i            ),
    .rst_ni           ( rst_ni           ),
    .test_mode_i      ( test_mode_i      ),
    .clear_i          ( clear_o          ),
    .ctrl_streamer_o  ( ctrl_streamer_o  ),
    .flags_streamer_i ( flags_streamer_i ),
    .ctrl_engine_o    ( ctrl_engine_o    ),
    .flags_engine_i   ( flags_engine_i   ),
    .ctrl_uloop_o     ( uloop_ctrl       ),
    .flags_uloop_i    ( uloop_flags      ),
    .ctrl_ctrlmult_o  ( ctrl_ctrlmult    ),
    .flags_ctrlmult_i ( flags_ctrlmult   ),
    .ctrl_slave_o     ( slave_ctrl       ),
    .flags_slave_i    ( slave_flags      ),
    .reg_file_i       ( reg_file         ),
    .ctrl_i           ( fsm_ctrl         )
  );


  ///////////
  // ULOOP //
  ///////////

  // uloop signals
  logic [RBE_ULOOP_NB_RO_REG-1:0][RBE_ULOOP_REG_WIDTH-1:0] uloop_registers_read;

  hwpe_ctrl_package::uloop_loops_t    [ RBE_ULOOP_NB_LOOPS-1:0] uloop_loops;
  hwpe_ctrl_package::uloop_bytecode_t [   RBE_ULOOP_LENGTH-1:0] uloop_bytecode;

  // get configured loops
  assign uloop_loops    = { '0, reg_file.hwpe_params[RBE_REG_UCODE_LOOPS1][31:0], reg_file.hwpe_params[RBE_REG_UCODE_LOOPS0]};
  // get configured bytecode
  assign uloop_bytecode = { '0, reg_file.hwpe_params[RBE_REG_UCODE_STATIC09], reg_file.hwpe_params[RBE_REG_UCODE_STATIC08:RBE_REG_UCODE_STATIC00] };

  // assign configurations to the uloop ctrl
  always_comb
  begin
    uloop_code = '0;
    for(int i=0; i<RBE_ULOOP_NB_LOOPS; i++) begin
      uloop_code.loops[i] = uloop_loops[i];
    end
    for(int i=0; i<RBE_ULOOP_LENGTH; i++) begin
      uloop_code.code [i] = uloop_bytecode[i];
    end
    uloop_code.range[0] = n_tiles_qa;
    uloop_code.range[1] = n_tiles_K_in;
    uloop_code.range[2] = n_tiles_K_out;
    uloop_code.range[3] = static_reg_n_xpatches_w;
    uloop_code.range[4] = static_reg_n_xpatches_h;
  end

  // compute and assign the uloop registers
  assign uloop_registers_read[RBE_ULOOP_MNEM_NIF]            = static_reg_kin;
  assign uloop_registers_read[RBE_ULOOP_MNEM_NOF]            = static_reg_kout;
  assign uloop_registers_read[RBE_ULOOP_MNEM_PATCH_SIZE]     = TP*static_reg_w<<2;
  assign uloop_registers_read[RBE_ULOOP_MNEM_PATCH_SIZE_W]   = TP*3;
  assign uloop_registers_read[RBE_ULOOP_MNEM_PATCH_SIZE_H]   = static_w_tiles_K_in_qa*TP*BUFFER_PATCH_SIZE_MIN<<2;
  assign uloop_registers_read[RBE_ULOOP_MNEM_PATCH_OUTPUT]   = n_tiles_qa_out=={'0,2'b10} ? TP*static_reg_ow<<3 : TP*static_reg_ow<<2;
  assign uloop_registers_read[RBE_ULOOP_MNEM_PATCH_OUTPUT_W] = TP*3;
  assign uloop_registers_read[RBE_ULOOP_MNEM_PATCH_OUTPUT_H] = patch_output_h;
  assign uloop_registers_read[RBE_ULOOP_MNEM_STREAM_SIZE]    = stream_size;
  assign uloop_registers_read[RBE_ULOOP_MNEM_NQ_BATCH]       = 32*(32+16);
  assign uloop_registers_read[RBE_ULOOP_MNEM_ZERO]           = '0;

  // uloop control
  hwpe_ctrl_uloop #(
    .LENGTH    ( RBE_ULOOP_LENGTH    ),
    .NB_LOOPS  ( RBE_ULOOP_NB_LOOPS  ),
    .NB_RO_REG ( RBE_ULOOP_NB_RO_REG ),
    .NB_REG    ( RBE_ULOOP_NB_REG    ),
    .REG_WIDTH ( RBE_ULOOP_REG_WIDTH ),
    .CNT_WIDTH ( RBE_ULOOP_CNT_WIDTH ),
    .SHADOWED  ( RBE_ULOOP_SHADOWED  )
  ) i_uloop (
    .clk_i            ( clk_i                ),
    .rst_ni           ( rst_ni               ),
    .test_mode_i      ( test_mode_i          ),
    .clear_i          ( clear_o              ),
    .ctrl_i           ( uloop_ctrl           ),
    .flags_o          ( uloop_flags          ),
    .uloop_code_i     ( uloop_code           ),
    .registers_read_i ( uloop_registers_read )
  );


  ////////////////////////////
  // SEQUENTIAL MULTIPLIERS //
  ////////////////////////////

  // signals
  logic [QUANT_CNT_SIZE+$clog2(TP)-1:0] static_reg_qw_shifted;
  logic [  QUANT_CNT_SIZE+$clog2(TP):0] static_reg_qw_shifted_special;
  logic [QUANT_CNT_SIZE+$clog2(TP)-1:0] static_reg_qw_shifted_bytes;
  logic [               $clog2(TP)+1:0] static_tp_2;

  logic mult_start;
  logic mult_valid;

  // simple precomputations
  assign static_reg_qw_shifted         = static_reg_qw << $clog2(TP);
  assign static_reg_qw_shifted_special = static_reg_qw_shifted + 2;
  assign static_reg_qw_shifted_bytes   = static_reg_qw << 2;
  assign static_tp_2                   = {'0, TP} + 2;

  hwpe_ctrl_seq_mult #(
    .AW( QUANT_CNT_SIZE+$clog2(TP) ),
    .BW( $clog2(TP)+2              )
  ) i_static_fs0_qw_mult (
    .clk_i  ( clk_i                       ),
    .rst_ni ( rst_ni                      ),
    .clear_i( clear_o                     ),
    .start_i( mult_start                  ),
    .a_i    ( static_reg_qw_shifted_bytes ),
    .b_i    ( static_tp_2                 ),
    .valid_o( static_fs0_qw_valid         ),
    .prod_o ( static_fs0_qw               )
  );

  hwpe_ctrl_seq_mult #(
    .AW( QUANT_CNT_SIZE+$clog2(TP)+1 ),
    .BW( RBE_ULOOP_REG_WIDTH/2       )
  ) i_static_fs0_fs0_qw_TP2_mult (
    .clk_i  ( clk_i                         ),
    .rst_ni ( rst_ni                        ),
    .clear_i( clear_o                       ),
    .start_i( mult_start                    ),
    .a_i    ( static_reg_qw_shifted_special ),
    .b_i    ( static_reg_fs0_fs0_qw         ),
    .valid_o( static_fs0_fs0_qw_TP2_valid   ),
    .prod_o ( static_fs0_fs0_qw_TP2         )
  );

  hwpe_ctrl_seq_mult #(
    .AW( SPATIAL_CNT_SIZE      ),
    .BW( RBE_ULOOP_CNT_WIDTH+2 )
  ) i_static_w_tiles_qa_k_in_mult (
    .clk_i  ( clk_i                        ),
    .rst_ni ( rst_ni                       ),
    .clear_i( clear_o                      ),
    .start_i( mult_start                   ),
    .a_i    ( static_reg_w                 ),
    .b_i    ( static_tiles_K_in_qa         ),
    .valid_o( static_w_tiles_K_in_qa_valid ),
    .prod_o ( static_w_tiles_K_in_qa       )
  );

  hwpe_ctrl_seq_mult #(
    .AW( SPATIAL_CNT_SIZE      ),
    .BW( RBE_ULOOP_CNT_WIDTH+2 )
  ) i_static_w_tiles_qa_out_k_out_mult (
    .clk_i  ( clk_i                             ),
    .rst_ni ( rst_ni                            ),
    .clear_i( clear_o                           ),
    .start_i( mult_start                        ),
    .a_i    ( static_reg_ow                     ),
    .b_i    ( static_tiles_K_out_qa_out         ),
    .valid_o( static_w_tiles_K_out_qa_out_valid ),
    .prod_o ( static_w_tiles_K_out_qa_out       )
  );

  assign mult_valid           = static_fs0_fs0_qw_TP2_valid & static_fs0_qw_valid  & static_w_tiles_K_in_qa_valid & static_w_tiles_K_out_qa_out_valid;
  assign flags_ctrlmult.valid = mult_valid;
  assign mult_start           = ctrl_ctrlmult.start;

endmodule // rbe_ctrl
