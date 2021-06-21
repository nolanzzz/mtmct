
import os
import pandas as pd



def arrange_single_camera_evaluation(evaluation_result_folder):
    evaluation_result_folders = os.listdir(evaluation_result_folder)

    all_eval_results = pd.DataFrame()

    for eval_res_folder in evaluation_result_folders:

        if eval_res_folder.endswith(".csv"):
            continue

        eval_results_path = os.path.join(evaluation_result_folder
                                        ,eval_res_folder
                                         ,"single_cam_evaluation"
                                         ,"chunks_mean_std_evaluation.csv")



        one_eval_result = pd.read_csv(eval_results_path)

        one_eval_result["Method"] = os.path.basename(eval_res_folder)

        one_eval_result = one_eval_result[one_eval_result["type"] == "mean"]
        eval_columns = ["Method", 'cam_id', 'IDF1', 'IDP', 'IDR', 'Rcll', 'Prcn'
            , 'GT', 'MT', 'PT', 'ML',
                        'FP', 'FN', 'IDs', 'FM', 'MOTA', 'MOTP']



        one_eval_result = one_eval_result[eval_columns]

        one_eval_result = one_eval_result.astype({"cam_id": int
                                               , 'GT': int
                                               , 'MT': int
                                               , 'PT': int
                                               , 'ML': int
                                               , "FP": int
                                               , "FN": int
                                               , "IDs": int
                                               , "FM": int
                                            })


        all_eval_results = all_eval_results.append(one_eval_result,ignore_index=True)

    all_eval_results = all_eval_results.sort_values(by=["cam_id","Method"])

    all_eval_results = all_eval_results.rename(columns={"cam_id": "Cam ID"})

    decimals = pd.Series(2, index=["IDF1"
                                        , "IDP"
                                        ,"IDR"
                                        ,"Rcll"
                                        , "Prcn"
                                        ,"MOTA"
                                        ,"MOTP"])

    all_eval_results = all_eval_results.round(decimals=decimals)

    all_eval_results = all_eval_results.replace(regex={r'^.*just_reid.*$': 'Only AF'
                                                        , r'^.*no_hom.*$': 'No HM'
                                                       , r'^.*all.*$': 'All'
                                                       , r'^.*no_muldis.*$': 'No MCTC'
                                                       , r'^.*no_onedis.*$': 'No SCTC'
                                                       , r'^.*no_pred.*$': 'No LP'
                                                       , r'^.*none.*$': 'None'})

    output_path = os.path.join(evaluation_result_folder,"single_cam_evaluation_arranged.csv")
    all_eval_results.to_csv(output_path,index=False)
    print(all_eval_results.to_string())




if __name__ == "__main__":
    arrange_single_camera_evaluation("/home/philipp/Dokumente/masterarbeit/JTA-MTMCT-Mod/thesistemplate/thesis/tables/evaluation_results_6.12")