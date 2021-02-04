/*
 * rbe_binconv_array.sv
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


module rbe_binconv_array #(
  parameter  int unsigned BC_COLUMN_SIZE    = rbe_package::BINCONV_COLUMN_SIZE, // number of BinConv blocks per column (default 9)
  parameter  int unsigned BC_NR_COLUMN      = rbe_package::BINCONV_COLUMN_SIZE, // number of BinConv columns (default 9 -- same of size of BinConv columns!)
  parameter  int unsigned BC_BLOCK_SIZE     = rbe_package::BINCONV_BLOCK_SIZE,  // number of SoP's per BinConv block (default 4)
  parameter  int unsigned BC_NR_ACTIVATIONS = 100,                              // 25 * BC_BLOCK_SIZE, FIXME
  parameter  int unsigned TP                = rbe_package::BINCONV_TP,          // number of input elements processed per cycle
  localparam int unsigned POPCOUNT_SIZE     = $clog2(TP)+1,
  localparam int unsigned BLOCK_PRES_SIZE   = POPCOUNT_SIZE+8+$clog2(BC_BLOCK_SIZE),
  localparam int unsigned COLUMN_PRES_SIZE  = BLOCK_PRES_SIZE+$clog2(BC_COLUMN_SIZE)
) (
  // global signals
  input  logic                              clk_i,
  input  logic                              rst_ni,
  input  logic                              test_mode_i,
  // local enable & clear
  input  logic                              enable_i,
  input  logic                              clear_i,
  // input activation stream + handshake
  hwpe_stream_intf_stream.sink              activation_i [BC_NR_ACTIVATIONS-1:0],
  // input weight stream + handshake
  hwpe_stream_intf_stream.sink              weight_i     [   BC_COLUMN_SIZE-1:0],
  // output features + handshake
  hwpe_stream_intf_stream.source            pres_o       [     BC_NR_COLUMN-1:0],
  // control and flags
  input  rbe_package::ctrl_binconv_array_t  ctrl_i,
  output rbe_package::flags_binconv_array_t flags_o
);

  /////////////
  // SIGNALS //
  /////////////

  logic [TP-1  :0] activation_data  [BC_NR_ACTIVATIONS-1:0];
  logic            activation_valid [BC_NR_ACTIVATIONS-1:0];
  logic [TP/8-1:0] activation_strb  [BC_NR_ACTIVATIONS-1:0];

  logic [TP-1  :0] activation_mapped_fs1_data  [BC_NR_COLUMN-1:0][BC_COLUMN_SIZE-1:0][BC_BLOCK_SIZE-1:0];
  logic            activation_mapped_fs1_valid [BC_NR_COLUMN-1:0][BC_COLUMN_SIZE-1:0][BC_BLOCK_SIZE-1:0];
  logic [TP/8-1:0] activation_mapped_fs1_strb  [BC_NR_COLUMN-1:0][BC_COLUMN_SIZE-1:0][BC_BLOCK_SIZE-1:0];

  logic [TP-1  :0] activation_mapped_fs3_data  [BC_NR_COLUMN-1:0][BC_COLUMN_SIZE-1:0][BC_BLOCK_SIZE-1:0];
  logic            activation_mapped_fs3_valid [BC_NR_COLUMN-1:0][BC_COLUMN_SIZE-1:0][BC_BLOCK_SIZE-1:0];
  logic [TP/8-1:0] activation_mapped_fs3_strb  [BC_NR_COLUMN-1:0][BC_COLUMN_SIZE-1:0][BC_BLOCK_SIZE-1:0];

  // interfaces
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( TP )
  ) activation_mapped [BC_NR_COLUMN*BC_COLUMN_SIZE*BC_BLOCK_SIZE-1:0] (.clk(clk_i));

`ifndef SYNTHESIS
  hwpe_stream_intf_stream #(
    .DATA_WIDTH       ( TP   ),
    .BYPASS_VCR_ASSERT( 1'b1 ),
    .BYPASS_VDR_ASSERT( 1'b1 )
  ) weight_int [BC_NR_COLUMN*BC_COLUMN_SIZE-1:0] (.clk(clk_i));
`else
  hwpe_stream_intf_stream #(
    .DATA_WIDTH ( TP )
  ) weight_int [BC_NR_COLUMN*BC_COLUMN_SIZE-1:0] (.clk(clk_i));
`endif

  //////////////////////////////////////
  // COLUMN, ROW AND BLOCK GENERATION //
  //////////////////////////////////////

  // weight ready assignment
  for (genvar jj=0; jj<BC_COLUMN_SIZE; jj++) begin
    assign weight_i[jj].ready = weight_int[jj].ready;
  end

  // activation extraction from interface
  for(genvar jj=0; jj<BC_NR_ACTIVATIONS; jj++) begin : activation_assignment_gen
    assign activation_data[jj]    = activation_i[jj].data;
    assign activation_valid[jj]   = activation_i[jj].valid;
    assign activation_strb[jj]    = activation_i[jj].strb;
    assign activation_i[jj].ready = activation_mapped[0].ready;
  end

  for(genvar ii=0; ii<BC_NR_COLUMN; ii++) begin : column_gen

    // weight assignment
    for (genvar jj=0; jj<BC_COLUMN_SIZE; jj++) begin : row_w_gen
      localparam ii_jj = ii*BC_COLUMN_SIZE+jj;
      assign weight_int[ii_jj].data  = weight_i[jj].data;
      assign weight_int[ii_jj].strb  = weight_i[jj].strb;
      assign weight_int[ii_jj].valid = weight_i[jj].valid;
    end // row_w_gen

    // pixel layout:
    //  0  4  8 12 16
    // 20 24 28 32 36
    // 40 44 48 52 56
    // 60 64 68 72 76
    // 80 84 88 92 96

    for(genvar rr=0; rr<BC_COLUMN_SIZE; rr++) begin : row_a_gen

      localparam i=0;
      localparam j=0;

      // filter size = 1
      // Todo: maybe different mapping for better useage of n_tiles_Kin > 1 ???
      localparam j_fs1  = (ii % 3);
      localparam i_fs1  = ((ii-(ii%3)) / 3);

      // filter size = 3
      localparam fj_fs3 = rr % 3;
      localparam fi_fs3 = (rr-fj_fs3) / 3;
      localparam j_fs3  = ii % 3 + fj_fs3;
      localparam i_fs3  = (ii-(ii%3)) / 3 + fi_fs3;

      for(genvar bb=0; bb<BC_BLOCK_SIZE; bb++) begin : block_gen

        // TODO: parameterize 5 (also BC_NR_ACTIVATIONS)
        assign activation_mapped_fs3_data [ii][rr][bb][TP-1:0]   = activation_data [i_fs3*4*5 + j_fs3*4 + bb][TP-1:0];
        assign activation_mapped_fs3_valid[ii][rr][bb]           = activation_valid[i_fs3*4*5 + j_fs3*4 + bb];
        assign activation_mapped_fs3_strb [ii][rr][bb][TP/8-1:0] = activation_strb [i_fs3*4*5 + j_fs3*4 + bb][TP/8-1:0];

        assign activation_mapped_fs1_data [ii][rr][bb][TP-1:0]   = activation_data [i_fs1*4*5 +
                                                                    j_fs1*4 + bb][TP-1:0];
        assign activation_mapped_fs1_valid[ii][rr][bb]           = activation_valid[i_fs1*4*5 +
                                                                    j_fs1*4 + bb];
        assign activation_mapped_fs1_strb [ii][rr][bb][TP/8-1:0] = activation_strb [i_fs1*4*5 + j_fs1*4 + bb][TP/8-1:0];

        localparam ii_rr_bb = ii*(BC_COLUMN_SIZE*BC_BLOCK_SIZE) + rr*(BC_BLOCK_SIZE) + bb;

        assign activation_mapped[ii_rr_bb].valid = (ctrl_i.fs == 1) ?
                                                    activation_mapped_fs1_valid[ii][rr][bb] :
                                                    activation_mapped_fs3_valid[ii][rr][bb];
        assign activation_mapped[ii_rr_bb].data  = (ctrl_i.fs == 1) ?
                                                    activation_mapped_fs1_data[ii][rr][bb] :
                                                    activation_mapped_fs3_data[ii][rr][bb];
        assign activation_mapped[ii_rr_bb].strb  = (ctrl_i.fs == 1) ?
                                                    activation_mapped_fs1_strb[ii][rr][bb] :
                                                    activation_mapped_fs3_strb[ii][rr][bb];

      end //block_gen
    end // row_a_gen

    // index localparams
    localparam int unsigned A_UPPER_IDX = (ii+1)*BC_COLUMN_SIZE*BC_BLOCK_SIZE;
    localparam int unsigned A_LOWER_IDX = (ii  )*BC_COLUMN_SIZE*BC_BLOCK_SIZE;

    localparam int unsigned W_UPPER_IDX = (ii+1)*BC_COLUMN_SIZE;
    localparam int unsigned W_LOWER_IDX = (ii  )*BC_COLUMN_SIZE;

    // column instantiation
    rbe_binconv_column #(
      .BC_COLUMN_SIZE ( BC_COLUMN_SIZE ),
      .BC_BLOCK_SIZE  ( BC_BLOCK_SIZE  ),
      .TP             ( TP             )
    ) i_column (
      .clk_i         ( clk_i                                        ),
      .rst_ni        ( rst_ni                                       ),
      .test_mode_i   ( test_mode_i                                  ),
      .enable_i      ( enable_i                                     ),
      .clear_i       ( clear_i                                      ),
      .activation_i  ( activation_mapped[A_UPPER_IDX-1:A_LOWER_IDX] ),
      .weight_i      ( weight_int[W_UPPER_IDX-1:W_LOWER_IDX]        ),
      .column_pres_o ( pres_o[ii]                                   ),
      .ctrl_i        ( ctrl_i.ctrl_column                           ),
      .flags_o       ( flags_o.flags_column[ii]                     )
    );

  end // column_gen

endmodule // rbe_binconv_array

