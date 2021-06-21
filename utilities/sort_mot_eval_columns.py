import pandas as pd




def sort_mot_eval_columns(in_path_or_dataframe,out_path):


    if isinstance(in_path_or_dataframe,str):
        in_path_or_dataframe = pd.read_csv(in_path_or_dataframe)

    column_names_sorted = ["Method", 'IDF1', 'IDP', 'IDR', 'Rcll', 'Prcn'
                            , 'GT', 'MT', 'PT', 'ML'
                            , 'FP', 'FN', 'IDs', 'FM', 'MOTA', 'MOTP']

    in_path_or_dataframe = in_path_or_dataframe[column_names_sorted]

    in_path_or_dataframe.to_csv(out_path)






if __name__ == "__main__":
    sort_mot_eval_columns(
        in_path_or_dataframe="/media/philipp/philippkoehl_ssd/work_dirs/evaluation_results_mta/MTA_FRCNN.csv"
        , out_path="/media/philipp/philippkoehl_ssd/work_dirs/evaluation_results_mta/MTA_FRCNN_sorted.csv")