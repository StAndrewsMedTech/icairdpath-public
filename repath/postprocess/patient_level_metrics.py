from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from repath.utils.metrics import conf_mat_raw, save_conf_mat_plot, save_conf_mat_plot_ci


def compute_patient_level_scores(slide_results: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        slide_results:
    Returns:
        Data frame with two columns, patient and stage_label
    """
    tags = slide_results.tags
    patient_names = [tag.split(';')[0] for tag in tags]
    patient_label = [tag.split(';')[1] for tag in tags]
    patient_label = np.array(patient_label)
    slide_results['patient'] = patient_names
    unique_patients = slide_results.patient.unique()
    scores = []
    unique_patient_labels = []
    for patient in unique_patients:
        slides_for_patient = slide_results.predictions[slide_results.patient == patient]
        assert len(slides_for_patient) == 5, f'Wrong number of slides for patient found {patient}'
        num_negative = np.sum(slides_for_patient == 'negative')
        num_micro = np.sum(slides_for_patient == 'micro')
        num_macro = np.sum(slides_for_patient == 'macro')
        num_itc = np.sum(slides_for_patient == 'itc')
        label_for_patients = patient_label[slide_results.patient == patient]
        unique_patient_labels.append(label_for_patients[0])

        # PN0: Nomicro - metastases or macro - metastases or ITCs found.
        if num_negative == 5:
            patient_score = 'pN0'
        else:
            # pN0(i +): Only ITCs found.
            if num_negative + num_itc == 5:
                patient_score = 'pN0(i+)'
            else:
                # pN1mi: Micro - metastases found, but nomacro - metastases found.
                if num_negative + num_itc + num_micro == 5:
                    patient_score = 'pN1mi'
                else:
                    # pN1: Metastases found in 1–3 lymph nodes, of which at least one is a macro - metastasis.
                    num_metastases = num_micro + num_macro
                    if 1 <= num_metastases <= 3 and num_macro >= 1:
                        patient_score = 'pN1'

                    # pN2: Metastases found in 4–9 lymph nodes, of which at least one is a macro - metast
                    if num_metastases > 3 and num_macro >= 1:
                        patient_score = 'pN2'

        scores.append(patient_score)

    unique_patients = pd.DataFrame(unique_patients, columns=['patient'])
    unique_patients['predicted_stage'] = scores
    unique_patients['true_stage'] = unique_patient_labels

    return unique_patients


def calculate_kappa(reference, submission):
    """
    Calculate inter-annotator agreement with quadratic weighted Kappa.
    Args:
        reference (pandas.DataFrame): List of labels assigned by the organizers.
        submission (pandas.DataFrame): List of labels assigned by participant.
    Returns:
        float: Kappa score.
    Raises:
        ValueError: Unknown stage in reference.
        ValueError: Patient missing from submission.
        ValueError: Unknown stage in submission.
    """

    # The accepted stages are pN0, pN0(i+), pN1mi, pN1, pN2 as described on the website. During parsing all strings converted to lowercase.
    #
    stage_list = ['pn0', 'pn0(i+)', 'pn1mi', 'pn1', 'pn2']

    # Extract the patient pN stages from the tables for evaluation.
    #
    reference_map = {df_row[0]: df_row[1].lower() for _, df_row in reference.iterrows() if df_row[0].lower().endswith('.zip')}
    submission_map = {df_row[0]: df_row[1].lower() for _, df_row in submission.iterrows() if df_row[0].lower().endswith('.zip')}

    # Reorganize data into lists with the same patient order and check consistency.
    #
    reference_stage_list = []
    submission_stage_list = []
    for patient_id, submission_stage in submission_map.items():
        # Check consistency: all stages must be from the official stage list and there must be a submission for each patient in the ground truth.
        
        submission_stage = submission_stage.lower()
        reference_stage = reference_map[patient_id].lower()

        if reference_stage not in stage_list:
            raise ValueError('Unknown stage in reference: \'{stage}\''.format(stage=reference_stage))
        if patient_id not in submission_map:
            raise ValueError('Patient missing from submission: \'{patient}\''.format(patient=patient_id))
        if submission_stage not in stage_list:
            raise ValueError('Unknown stage in submission: \'{stage}\''.format(stage=submission_map[patient_id]))

        # Add the pair to the lists.
        #
        reference_stage_list.append(reference_stage)
        submission_stage_list.append(submission_stage)

    # Return the Kappa score.
    #
    return cohen_kappa_score(y1=reference_stage_list, y2=submission_stage_list, labels=stage_list, weights='quadratic')


def calc_patient_level_metrics(input_dir: Path, output_dir: Path, title: str, ci=True, nreps=1000) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    slide_results = pd.read_csv(input_dir / 'slide_results.csv')

    unique_patients = compute_patient_level_scores(slide_results)
    unique_patients.to_csv(input_dir / 'patient_results.csv', index=False)

    kappa = cohen_kappa_score(y1=unique_patients['true_stage'].tolist(), y2=unique_patients['predicted_stage'].tolist(), weights='quadratic')

    patient_labels = ['pN0', 'pN0(i+)', 'pN1mi', 'pN1', 'pN2']
    conf_mat = conf_mat_raw(unique_patients.true_stage.to_numpy(), unique_patients.predicted_stage.to_numpy(), labels=patient_labels)
    conf_mat = conf_mat.ravel().tolist()
    pred_tiled_labels = patient_labels * len(patient_labels)
    true_tiled_labels = [item for item in patient_labels for i in range(len(patient_labels))]
    confmat_labels = [f'true_{vals[0]}_pred_{vals[1]}' for vals in list(zip(true_tiled_labels, pred_tiled_labels))]
    column_labels = ['kappa'] + confmat_labels

    output_list = [kappa] + conf_mat
    output_arr = np.reshape(np.array(output_list), (1, len(output_list)))
    patient_metrics_out = pd.DataFrame(output_arr)
    patient_metrics_out.columns = column_labels
    patient_metrics_out.index = ['results']

     # create confidence matrix plot and write out
    title_cm = "Patient Classification Confusion Matrix for \n" + title
    title_plus_kappa = title_cm + "\n kappa = " + str(round(kappa, 2))
    save_conf_mat_plot(patient_metrics_out.iloc[:, 1:], patient_labels, title_plus_kappa, output_dir)

    kappa_samples = []
    if ci:
        kappa1000 = np.empty((nreps, 1))
        conf_mat1000 = np.empty((nreps, len(patient_labels)**2))
        for rep in range(nreps):
            sample_patients = unique_patients.sample(frac=1.0, replace=True)
            sample_kappa = cohen_kappa_score(y1=sample_patients['true_stage'].tolist(), y2=sample_patients['predicted_stage'].tolist(), weights='quadratic')
            kappa1000[rep, 0] = sample_kappa
            conf_mat = conf_mat_raw(sample_patients.true_stage.to_numpy(), sample_patients.predicted_stage.to_numpy(), labels=patient_labels)
            conf_mat = conf_mat.ravel().tolist()
            conf_mat1000[rep, :] = conf_mat

        samples_df = pd.DataFrame(np.hstack((kappa1000, conf_mat1000)), columns=column_labels)
        samples_df.index = ['sample_' + str(x) for x in range(nreps)]
        patient_metrics_ci = samples_df.quantile([0.025, 0.975])
        patient_metrics_ci.index = ['ci_lower_bound', 'ci_upper_bound']
        patient_metrics_out = pd.concat((patient_metrics_out, patient_metrics_ci, samples_df), axis=0)

        title_with_ci = title_cm + "\n kappa = " + str(round(kappa, 2)) + ", (" + str(round(patient_metrics_ci.iloc[0, 0], 2)) + ", " + str(round(patient_metrics_ci.iloc[0, 0], 2)) + ")" 
        # create confidence matrix plot with confidence interval and write out
        save_conf_mat_plot_ci(patient_metrics_out.iloc[:, 1:], patient_labels, title_with_ci, output_dir)

    patient_metrics_out.to_csv(output_dir / 'patient_metrics.csv')



