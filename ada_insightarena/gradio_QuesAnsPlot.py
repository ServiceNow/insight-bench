import gradio as gr
import json
import os
import random
import uuid
from datetime import datetime

###############################################################################
# 1) GLOBAL CONFIG
###############################################################################
RESULTS_FILE = "results/merged_pilot2ndrun_results.json"  # Merged file with exp_group, hash, questions
SAVE_DIR = "results/human_eval_pilot2nd"                  # Where to save final JSONs

###############################################################################
# 2) LOAD METADATA & GOAL
###############################################################################
def load_metadata(dataset_id):
    meta_path = f"data/jsons/{dataset_id}/meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_goaldata(dataset_id):
    goal_path = f"data/jsons/{dataset_id}/goal.json"
    if not os.path.exists(goal_path):
        return {}
    with open(goal_path, "r", encoding="utf-8") as f:
        return json.load(f)

###############################################################################
# 3) FORMAT DATA (Dataset Info + Goal/Persona)
###############################################################################
def format_dataset_info(metadata):
    domain = str(metadata.get("domain") or "N/A")
    desc   = str(metadata.get("dataset_description") or "N/A")
    urls   = metadata.get("dataset_urls") or []
    if not isinstance(urls, list):
        urls = []
    nb_url = str(metadata.get("notebook_url") or "N/A")

    md = "## 📊 Dataset Information\n"
    md += f"🏷️ **Domain:** {domain}\n\n"
    md += "📝 **Dataset Description:**\n" + desc + "\n\n"
    md += "🔗 **Dataset URLs:**\n"
    for u in urls:
        md += f"- {u}\n"
    md += f"\n📓 **Notebook URL:** {nb_url}\n\n"
    return md

def format_goal_persona(goal):
    goal_text = str(goal.get("goal") or "N/A")
    persona_text = str(goal.get("persona") or "N/A")
    return f"## 🎯 Goal\n{goal_text}\n\n## 🧑 Persona\n{persona_text}\n\n"

###############################################################################
# 4) BOLDIFY HEADINGS
###############################################################################
def boldify_headings(text):
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("### "):
            heading = line[4:].strip()
            new_lines.append(f"<span style='font-weight:bold; color:black;'>{heading}</span>")
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

###############################################################################
# 5) LOAD RESPONSES
###############################################################################
def load_responses(dataset_id):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    key = f"dataset_{dataset_id}"
    dataset_obj = all_data.get(key, {})

    method_1 = dataset_obj.get("method_1", {})
    method_2 = dataset_obj.get("method_2", {})

    return {
        "method_1": {
            "exp_group": method_1.get("exp_group", "no_exp_group"),
            "hash": method_1.get("hash", "no_hash"),
            "questions": method_1.get("questions", {})
        },
        "method_2": {
            "exp_group": method_2.get("exp_group", "no_exp_group"),
            "hash": method_2.get("hash", "no_hash"),
            "questions": method_2.get("questions", {})
        }
    }

###############################################################################
# 6) COMPARE OUTPUTS
###############################################################################
def compare_outputs(dataset_id, question_idx=0):
    metadata = load_metadata(dataset_id)
    goal = load_goaldata(dataset_id)
    dataset_md = format_dataset_info(metadata)
    goal_md = format_goal_persona(goal)

    resp = load_responses(dataset_id)
    m1q = resp["method_1"]["questions"].get(str(question_idx), {})
    m2q = resp["method_2"]["questions"].get(str(question_idx), {})

    question_text = m1q.get("question", "") or m2q.get("question", "")
    plot_a = m1q.get("plot_path", "")
    insight_a = boldify_headings(m1q.get("insight", ""))
    plot_b = m2q.get("plot_path", "")
    insight_b = boldify_headings(m2q.get("insight", ""))

    return dataset_md, goal_md, question_text, plot_a, plot_b, insight_a, insight_b

###############################################################################
# 7) SAVE EVALUATION
###############################################################################
def save_evaluation(
    ds_id, q_idx,
    depth_sel, depth_comment,
    relevance_sel, relevance_comment,
    persona_sel, persona_comment,
    coherence_sel, coherence_comment,
    adequate_sel, adequate_comment,
    plot_sel, plot_comment,
    exp_group_a, hash_a, skill_a,
    exp_group_b, hash_b, skill_b,
    output_a, output_b,
    designation,
    user_id
):
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rubrics = {
        "depth_of_analysis": {
            "selection": depth_sel,
            "comment": depth_comment
        },
        "relevance_to_goal": {
            "selection": relevance_sel,
            "comment": relevance_comment
        },
        "persona_consistency": {
            "selection": persona_sel,
            "comment": persona_comment
        },
        "coherence": {
            "selection": coherence_sel,
            "comment": coherence_comment
        },
        "answers_question_adequately": {
            "selection": adequate_sel,
            "comment": adequate_comment
        },
        "plot_conclusion": {
            "selection": plot_sel,
            "comment": plot_comment
        }
    }

    evaluation_data = {
        "dataset_id": ds_id,
        "question_idx": q_idx,
        "timestamp": timestamp,
        "designation": designation,
        "user_id": user_id,
        "rubrics": rubrics,
        "model_a": {
            "exp_group": exp_group_a,
            "hash": hash_a,
            "skill": skill_a,
            "output": output_a
        },
        "model_b": {
            "exp_group": exp_group_b,
            "hash": hash_b,
            "skill": skill_b,
            "output": output_b
        }
    }

    filename = f"{SAVE_DIR}/{ds_id}_{q_idx}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2, ensure_ascii=False)

    return f"✅ Evaluation saved to {filename}"

###############################################################################
# 8) GET ALL DATASET IDS
###############################################################################
def get_all_datasets_from_json():
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    ds_list = []
    for k in all_data.keys():
        if k.startswith("dataset_"):
            try:
                ds_num = int(k.split("_")[1])
                ds_list.append(ds_num)
            except:
                pass
    ds_list.sort()
    return ds_list

def get_next_dataset_id(current_id):
    ds_list = get_all_datasets_from_json()
    if current_id not in ds_list:
        return None
    idx = ds_list.index(current_id)
    if idx < len(ds_list) - 1:
        return ds_list[idx + 1]
    else:
        return None

def get_prev_dataset_id(current_id):
    ds_list = get_all_datasets_from_json()
    if current_id not in ds_list:
        return None
    idx = ds_list.index(current_id)
    if idx > 0:
        return ds_list[idx - 1]
    else:
        return None

def get_question_count(dataset_id):
    r = load_responses(dataset_id)
    q1 = r["method_1"]["questions"]
    q2 = r["method_2"]["questions"]
    return max(len(q1), len(q2))

###############################################################################
# 9) RANDOMIZE
###############################################################################
def update_method_info_fn(ds_id, question_idx=0):
    resp = load_responses(ds_id)
    m1 = resp["method_1"]
    m2 = resp["method_2"]

    q1 = m1["questions"].get(str(question_idx), {})
    q2 = m2["questions"].get(str(question_idx), {})

    skill_a = q1.get("skill", "no_skill")
    skill_b = q2.get("skill", "no_skill")

    pair = [
        (m1["exp_group"], m1["hash"], skill_a),
        (m2["exp_group"], m2["hash"], skill_b)
    ]
    random.shuffle(pair)

    md_a = "### Model A"
    md_b = "### Model B"

    exA, haA, skA = pair[0]
    exB, haB, skB = pair[1]

    return (md_a, md_b, exA, haA, skA, exB, haB, skB)

###############################################################################
# 10) BUILD THE GRADIO APP
###############################################################################
import gradio as gr

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏆 AdaAgent: Skill-Adaptive Data Analytics\nCompare outputs from two methods question-by-question, then evaluate them on **6 rubrics**!")
    gr.Markdown("---")

    designation_dropdown = gr.Dropdown(
        choices=["Choose here","data analyst", "data engineer", "data scientist", "statistician", "data science practitioner", "data science researcher"],
        label="User Designation",
        value="Choose here"
    )

    # Hidden dataset slider
    ds_list = get_all_datasets_from_json()
    dataset_slider = gr.Slider(
        minimum=ds_list[0] if ds_list else 0,
        maximum=ds_list[-1] if ds_list else 0,
        step=1,
        value=ds_list[0] if ds_list else 0,
        label="Dataset ID",
        visible=False
    )


    model_a_header = gr.Markdown("### Model A", visible=False)
    model_b_header = gr.Markdown("### Model B", visible=False)

    exp_group_a_state = gr.State()
    hash_a_state      = gr.State()
    skill_a_state     = gr.State()

    exp_group_b_state = gr.State()
    hash_b_state      = gr.State()
    skill_b_state     = gr.State()

    metadata_display = gr.Markdown()
    question_text    = gr.Textbox(label="Question")

    question_slider = gr.Slider(
        minimum=0,
        maximum=0,
        step=1,
        value=0,
        label="Question Index",
        visible=True
    )

    with gr.Row(equal_height=True):
        model_a_plot = gr.Image(label="Model A (Plot & Answer)")
        model_b_plot = gr.Image(label="Model B (Plot * Answer)")
        # model_a_insight = gr.Markdown(label="Model A Output")
        # model_b_insight = gr.Markdown(label="Model B Output")
    with gr.Row(equal_height=True):
        model_a_insight = gr.Markdown(label="Model A Output",visible=True)
        model_b_insight = gr.Markdown(label="Model B Output",visible=True)
    goal_persona_display = gr.Markdown(label="Goal & Persona")

    # designation_dropdown = gr.Dropdown(
    #     choices=["Choose here","data analyst", "data engineer", "data scientist", "statistician", "data science practitioner", "data science researcher"],
    #     label="Designation",
    #     value="Choose here"
    # )

    rubric_choices = ["A is better", "B is better", "Tie", "None are good"]

    depth_radio = gr.Radio(rubric_choices, label="🔍 1) Depth of Analysis", value=None,
            info=("<div style='background-color:#fff; color:#000; padding:8px; font-size:16px; line-height:1.4;'>"
                  "<strong>Measures how thoroughly the insight explores underlying patterns, relationships, and potential causes in the data.</strong> "
                  "A strong response goes beyond surface observations, incorporating relevant factors, comparisons, or statistical reasoning."
                  "</div>"))
    depth_comment = gr.Textbox(label="Depth Comment (optional)", lines=2)

    relevance_radio = gr.Radio(rubric_choices, label="🌟 2) Relevance to goal", value=None,
            info=("<div style='background-color:#fff; color:#000; padding:8px; font-size:16px; line-height:1.4;'>"
                  "<strong>Evaluates how closely the insight addresses the primary objective or question of the analysis.</strong> "
                  "A relevant response stays focused on the dataset’s context and the stated goal."
                  "</div>"))
    relevance_comment = gr.Textbox(label="Relevance Comment (optional)", lines=2)

    persona_radio = gr.Radio(rubric_choices, label="👤 3) Persona consistency", value=None,
            info=("<div style='background-color:#fff; color:#000; padding:8px; font-size:16px; line-height:1.4;'>"
                  "<strong>Checks if the insight is presented from the intended stakeholder or persona's perspective.</strong> "
                  "A consistent response frames findings and conclusions aligned with that persona’s role."
                  "</div>"))
    persona_comment = gr.Textbox(label="Persona consistency Comment (optional)", lines=2)

    coherence_radio = gr.Radio(rubric_choices, label="✂️ 4) Coherence", value=None,
            info=("<div style='background-color:#fff; color:#000; padding:8px; font-size:16px; line-height:1.4;'>"
                  "<strong>Evaluates the clarity, organization, and logical flow of the insight.</strong> "
                  "A coherent response is well-structured and uses clear language."
                  "</div>"))
    coherence_comment = gr.Textbox(label="Coherence Comment (optional)", lines=2)

    adequate_radio = gr.Radio(rubric_choices, label="❓ 5) Which model answers the question adequately?", value=None,
            info=("<div style='background-color:#fff; color:#000; padding:8px; font-size:16px; line-height:1.4;'>"
                  "<strong>Determines how effectively each model’s text addresses the question.</strong> "
                  "An adequate response is accurate, complete, and directly relevant."
                  "</div>"))
    adequate_comment = gr.Textbox(label="Adequacy Comment (optional)", lines=2)

    plot_radio = gr.Radio(rubric_choices, label="📊 6) Which model shows a proper conclusion of the plot?", value=None,
            info=("<div style='background-color:#fff; color:#000; padding:8px; font-size:16px; line-height:1.4;'>"
                  "<strong>Assesses how accurately each model interprets the plot and draws a meaningful conclusion.</strong> "
                  "A strong response references the visual data and explains its significance."
                  "</div>"))
    plot_comment = gr.Textbox(label="Plot Conclusion Comment (optional)", lines=2)


    # Buttons row: previous, submit, next
    with gr.Row():
        prev_button = gr.Button("Previous", variant="secondary")
        submit_button = gr.Button("Submit Rubrics", variant="primary")
        next_button = gr.Button("Next", variant="secondary")

    feedback = gr.Markdown()

    ###########################################################################
    # HELPER: question_slider range
    ###########################################################################
    def update_question_slider_fn(ds_id):
        c = get_question_count(ds_id)
        if c == 0:
            return gr.update(minimum=0, maximum=0, value=0)
        return gr.update(minimum=0, maximum=c-1, value=0)

    dataset_slider.change(fn=update_question_slider_fn, inputs=[dataset_slider], outputs=[question_slider])

    ###########################################################################
    # HELPER: update_display
    ###########################################################################
    def update_display_fn(ds_id, q_idx):
        ds_md, gp_md, q_text, pA, pB, insA, insB = compare_outputs(ds_id, q_idx)
        info = update_method_info_fn(ds_id, q_idx)
        _, _, exA, haA, skA, exB, haB, skB = info
        return (
            ds_md,         # metadata_display
            q_text,        # question_text
            pA,            # model_a_plot
            pB,            # model_b_plot
            insA,          # model_a_insight
            insB,          # model_b_insight
            gp_md,         # goal_persona_display
            exA, haA, skA, # exp_group_a_state, hash_a_state, skill_a_state
            exB, haB, skB  # exp_group_b_state, hash_b_state, skill_b_state
        )

    dataset_slider.change(
        fn=update_display_fn,
        inputs=[dataset_slider, question_slider],
        outputs=[
            metadata_display, question_text,
            model_a_plot, model_b_plot,
            model_a_insight, model_b_insight,
            goal_persona_display,
            exp_group_a_state, hash_a_state, skill_a_state,
            exp_group_b_state, hash_b_state, skill_b_state
        ]
    )
    question_slider.change(
        fn=update_display_fn,
        inputs=[dataset_slider, question_slider],
        outputs=[
            metadata_display, question_text,
            model_a_plot, model_b_plot,
            model_a_insight, model_b_insight,
            goal_persona_display,
            exp_group_a_state, hash_a_state, skill_a_state,
            exp_group_b_state, hash_b_state, skill_b_state
        ]
    )

    ###########################################################################
    # NAVIGATION: next, previous
    ###########################################################################
    def navigate_next(ds_id, q_idx):
        c = get_question_count(ds_id)
        if q_idx < c - 1:
            return (gr.update(value=q_idx+1), ds_id)
        else:
            # last question => next dataset
            nxt = get_next_dataset_id(ds_id)
            if nxt is None:
                return (gr.update(value=q_idx), ds_id)  # no next dataset
            # move to next dataset
            return (gr.update(value=0), nxt)

    def navigate_previous(ds_id, q_idx):
        if q_idx > 0:
            return (gr.update(value=q_idx-1), ds_id)
        else:
            # first question => previous dataset
            prv = get_prev_dataset_id(ds_id)
            if prv is None:
                return (gr.update(value=q_idx), ds_id)  # no prev dataset
            # move to last question of prev dataset
            c = get_question_count(prv)
            return (gr.update(value=c-1), prv)

    next_button.click(
        fn=navigate_next,
        inputs=[dataset_slider, question_slider],
        outputs=[question_slider, dataset_slider]
    )

    prev_button.click(
        fn=navigate_previous,
        inputs=[dataset_slider, question_slider],
        outputs=[question_slider, dataset_slider]
    )

    ###########################################################################
    # SUBMIT RUBRICS
    ###########################################################################
    def vote_fn(
        ds_id, q_idx,
        d_sel, d_com,
        r_sel, r_com,
        p_sel, p_com,
        co_sel, co_cmt,
        ad_sel, ad_cmt,
        pl_sel, pl_cmt,
        outA, outB,
        exA, haA, skA,
        exB, haB, skB,
        designation
    ):
        # check rubrics
        all_sel = [d_sel, r_sel, p_sel, co_sel, ad_sel, pl_sel]
        if all(s is None for s in all_sel):
            msg = "Evaluation skipped for this question."
        else:
            user_id = str(uuid.uuid4())
            msg = save_evaluation(
                ds_id, q_idx,
                d_sel, d_com,
                r_sel, r_com,
                p_sel, p_com,
                co_sel, co_cmt,
                ad_sel, ad_cmt,
                pl_sel, pl_cmt,
                exA, haA, skA,
                exB, haB, skB,
                outA, outB,
                designation,
                user_id
            )

        # auto-increment question
        c = get_question_count(ds_id)
        if q_idx < c - 1:
            new_q = q_idx + 1
            new_ds = ds_id
        else:
            # last question => next dataset if any
            nxt = get_next_dataset_id(ds_id)
            if nxt is None:
                new_q = q_idx
                new_ds = ds_id
            else:
                new_q = 0
                new_ds = nxt

        # Return 14 items to reset rubrics
        return (
            msg,
            gr.update(value=new_ds),           # dataset_slider
            gr.update(value=new_q),            # question_slider
            gr.update(value=None),             # depth_radio
            gr.update(value=""),               # depth_comment
            gr.update(value=None),             # relevance_radio
            gr.update(value=""),               # relevance_comment
            gr.update(value=None),             # persona_radio
            gr.update(value=""),               # persona_comment
            gr.update(value=None),             # coherence_radio
            gr.update(value=""),               # coherence_comment
            gr.update(value=None),             # adequate_radio
            gr.update(value=""),               # adequate_comment
            gr.update(value=None),             # plot_radio
            gr.update(value="")                # plot_comment
        )

    submit_button.click(
        fn=lambda ds, q,
                  d_s, d_c,
                  r_s, r_c,
                  p_s, p_c,
                  co_s, co_c,
                  ad_s, ad_c,
                  pl_s, pl_c,
                  outA, outB,
                  exA, haA, skA,
                  exB, haB, skB,
                  desg: vote_fn(
            ds, q,
            d_s, d_c,
            r_s, r_c,
            p_s, p_c,
            co_s, co_c,
            ad_s, ad_c,
            pl_s, pl_c,
            outA, outB,
            exA, haA, skA,
            exB, haB, skB,
            desg
        ),
        inputs=[
            dataset_slider, question_slider,
            depth_radio, depth_comment,
            relevance_radio, relevance_comment,
            persona_radio, persona_comment,
            coherence_radio, coherence_comment,
            adequate_radio, adequate_comment,
            plot_radio, plot_comment,
            model_a_insight, model_b_insight,
            exp_group_a_state, hash_a_state, skill_a_state,
            exp_group_b_state, hash_b_state, skill_b_state,
            designation_dropdown
        ],
        outputs=[
            # total 15 outputs
            feedback,             # 1
            dataset_slider,       # 2
            question_slider,      # 3
            depth_radio,          # 4
            depth_comment,        # 5
            relevance_radio,      # 6
            relevance_comment,    # 7
            persona_radio,        # 8
            persona_comment,      # 9
            coherence_radio,      # 10
            coherence_comment,    # 11
            adequate_radio,       # 12
            adequate_comment,     # 13
            plot_radio,           # 14
            plot_comment          # 15
        ]
    )

    ###########################################################################
    # ON LOAD
    ###########################################################################
    def init_slider(ds_id):
        return update_question_slider_fn(ds_id)

    def init_display(ds_id, q_idx):
        return update_display_fn(ds_id, q_idx)

    demo.load(fn=init_slider, inputs=[dataset_slider], outputs=[question_slider])
    demo.load(
        fn=init_display,
        inputs=[dataset_slider, question_slider],
        outputs=[
            metadata_display, question_text,
            model_a_plot, model_b_plot,
            model_a_insight, model_b_insight,
            goal_persona_display,
            exp_group_a_state, hash_a_state, skill_a_state,
            exp_group_b_state, hash_b_state, skill_b_state
        ]
    )

    # Optional custom CSS
    gr.Markdown("""
    <style>
    .vote-button {
        min-width: 150px;
    }
    </style>
    """)

demo.launch(share=True)



