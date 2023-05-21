import json
import os
from clevr.vqa_eval import VQAEval
from clevr.vqa import VQA

# clevr_path = "/nobackup/users/zfchen/zt/clevr/CLEVR_v1.0"
# result_file_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/result_file.json"
# ann_file_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/ann_file.json"
# ques_file_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/ques_file.json"
# output_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/output.json"
# result_file_path = "F:/work//Multimodal-GPT/result_file.json"
# ann_file_path = "F:/work//Multimodal-GPT/ann_file.json"
# ques_file_path = "F:/work//Multimodal-GPT/ques_file.json"
# output_path = "F:/work//Multimodal-GPT/output.json"

def get_clevr_question(path_info):
    # clevr_path = "F:/work/CLEVR/CLEVR_v1.0/"
    clevr_path = path_info["clevr_path"]
    question_path = os.path.join(clevr_path, "questions", "CLEVR_val_questions.json")
    with open(question_path, "r") as f:
        question_list = json.load(f)["questions"]
    res_list = []
    for question in question_list:
        res_list.append({
            "split": question["split"],
            "image_id": question["image_index"],
            "image_path": os.path.join(clevr_path, "images", question["split"], question["image_filename"]),
            "question": question["question"],
            "answer": question["answer"],
        })
    return res_list[0:100]


def generate_result_file(path_info):
    # output_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/output.json"
    output_path = path_info["output_path"]
    result_file_path = path_info["result_file_path"]
    result_list = []
    outputs = json.load(open(output_path))
    for i, output in enumerate(outputs):
        result_list.append({
            "answer": output["response"],
            "question_id": i,
        })
    json.dump(result_list, open(result_file_path, "w"), indent=2)


def generate_ann_file(path_info):
    # output_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/output.json"
    output_path = path_info["output_path"]
    ann_file_path = path_info["ann_file_path"]
    ann = {
        "info": "",
        "data_type": "",
        "data_subtype": "",
        "annotations": [],
        "license": "",
    }
    ann_list = []
    outputs = json.load(open(output_path))
    for i, output in enumerate(outputs):
        if "image_id" not in output:
            output["image_id"] = i
        answers = []
        for j in range(1,11):
            answers.append({"answer_id": j, "answer": output["answer"], "answer_confidence": "yes"})
        ann_list.append({
            "question_id": i,
            "image_id": output["image_id"],
            "question_type": "test-question",
            "answer_type": "other",
            "answers": answers,
            "multiple_choice_answer": output["answer"]
            })
    ann["annotations"] = ann_list
    json.dump(ann, open(ann_file_path, "w"), indent=2)


def generate_ques_file(path_info):
    # output_path = "/nobackup/users/zfchen/zt/Multimodal-GPT/output.json"
    output_path = path_info["output_path"]
    ques_file_path = path_info["ques_file_path"]
    ques = {
        "info": "",
        "task_type": "Open-Ended",
        "data_type": "",
        "data_subtype": "",
        "questions": [],
        "license": "",
    }
    ques_list = []
    outputs = json.load(open(output_path))
    for i, output in enumerate(outputs):
        if "image_id" not in output:
            output["image_id"] = i
        ques_list.append({
            "question_id": i,
            "image_id": output["image_id"],
            "question": output["question"],
            })
    ques["questions"] = ques_list
    json.dump(ques, open(ques_file_path, "w"), indent=2)


def eval_output(path_info):
    generate_result_file(path_info)
    generate_ann_file(path_info)
    generate_ques_file(path_info)
    ann_file_path = path_info["ann_file_path"]
    ques_file_path = path_info["ques_file_path"]
    result_file_path = path_info["result_file_path"]
    vqa = VQA(ann_file_path, ques_file_path)
    vqa_result = vqa.loadRes(
        resFile=result_file_path, quesFile=ques_file_path
    )
    vqa_scorer = VQAEval(vqa, vqa_result, n=2)
    print("Start VQA evaluation.")
    vqa_scorer.evaluate()

    # print accuracies
    overall_acc = vqa_scorer.accuracy["overall"]

    print("Overall Accuracy is: %.02f\n" % overall_acc)


def generate_output(path_info, response_list):
    output_path = path_info["output_path"]
    output_list = get_clevr_question(path_info)
    for i, response in enumerate(response_list):
        output_list[i]["response"] = response
    with open(output_path, "w") as f:
        json.dump(output_list, f, indent=2)


if __name__ == "__main__":
    print(get_clevr_question()[0:100])