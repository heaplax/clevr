import json
from clevr.load_clevr import get_clevr_random_question


def get_llava_question(path_info, output_path, nums=100):
    question_list = get_clevr_random_question(path_info, nums)
    llava_question_list = []
    for index, question in enumerate(question_list):
        conversations = []
        conversations.append({
            "from": "human",
            "value": f"{question['question']}\n<image>"
        })
        conversations.append({
            "from": "gpt",
            "value": question["answer"]
        })
        llava_question_list.append({
            "id": index,
            "image": question["image_filename"],
            "conversations": conversations,
        })
    with open(output_path, "w") as f:
        json.dump(llava_question_list, f, indent=2)


def get_llava_eval_question(path_info, output_path, nums=100):
    question_list = get_clevr_random_question(path_info, nums)
    llava_eval_question_list = []
    for index, question in enumerate(question_list):
        llava_eval_question_list.append({
            "question_id": index,
            "image": question["image_filename"],
            "text": question['question'],
            "category": "complex",
            "answer": question["answer"],
        })
    with open(output_path, "w") as f:
        for question in llava_eval_question_list:
            f.write(json.dumps(question) + "\n")


if __name__ == "__main__":
    path_info = {
        "clevr_path": "F:/work/clevr/CLEVR_v1.0",
        "result_file_path": "/nobackup/users/zfchen/zt/OFA_transformer/OFA/result_file_OFA-base.json",
        "ann_file_path": "/nobackup/users/zfchen/zt/OFA_transformer/OFA/ann_file_OFA-base.json",
        "ques_file_path": "/nobackup/users/zfchen/zt/OFA_transformer/OFA/ques_file_OFA-base.json",
        "output_path": "/nobackup/users/zfchen/zt/OFA_transformer/OFA/output_OFA-base.json",
    }
    nums = 10000
    output_path = "F:/work/clevr/llava_clevr_train10000.json"
    get_llava_question(path_info, output_path, nums)

    # output_path = "F:/work/clevr/llava_clevr_eval1000.jsonl"
    # get_llava_eval_question(path_info, output_path, nums)