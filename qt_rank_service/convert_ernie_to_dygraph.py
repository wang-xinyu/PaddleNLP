from paddlenlp.transformers.ernie.static_to_dygraph_params.match_static_to_dygraph import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--layer_num", type=int, default=24)
parser.add_argument(
    "--dygraph_params_save_path",
    type=str,
    default="./dygraph_model/ernie_v2_chn_large.pdparams", )
parser.add_argument("--static_params_dir", default="", type=str)
# yapf: enable
args = parser.parse_args()
if __name__ == "__main__":
    convert_parameter_name_dict = {}

    convert_parameter_name_dict = match_embedding_param(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_encoder_param(
        convert_parameter_name_dict, layer_num=args.layer_num)
    convert_parameter_name_dict = match_pooler_parameter(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_mlm_parameter(
        convert_parameter_name_dict)
    convert_parameter_name_dict = match_last_fc_parameter(
        convert_parameter_name_dict)
    static_to_dygraph_param_name = {
        value: key
        for key, value in convert_parameter_name_dict.items()
    }

    for i, (static_name,
            dygraph_name) in enumerate(static_to_dygraph_param_name.items()):
        print("{}. {}:-------:{}".format(i, static_name, dygraph_name))

    convert_static_to_dygraph_params(
        dygraph_params_save_path=args.dygraph_params_save_path,
        static_params_dir=args.static_params_dir,
        static_to_dygraph_param_name=static_to_dygraph_param_name,
        model_name='ernie')
