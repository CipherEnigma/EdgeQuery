"""Generate the mid-term BTP report as a Word document matching the sample format."""
import os
import latex2mathml.converter
from lxml import etree
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# ── OMML equation helper ──────────────────────────────────────────────────────
XSLT_PATH = r"C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL"
_xslt = etree.parse(XSLT_PATH)
_transform = etree.XSLT(_xslt)

def _latex_to_omml(latex_str):
    mathml = latex2mathml.converter.convert(latex_str)
    omml = _transform(etree.fromstring(mathml.encode()))
    return omml.getroot()

def add_equation(doc, latex_str, label=None):
    """Insert a centred display equation with optional (3.1) label."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(8)
    omml_el = _latex_to_omml(latex_str)
    p._element.append(omml_el)
    if label:
        r = p.add_run(f"     {label}")
        r.font.size = Pt(12)
        r.font.name = 'Calibri'
    return p

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "BTP_Midterm_Report_Pragunie_Aditya_v5.docx")
SIG = os.path.join(HERE, "figures", "signature_pragunie.jpg")
SIG_SUPERVISOR = os.path.join(HERE, "figures", "signature_suchi_kumari.jpg")
FIG_PIPELINE = os.path.join(HERE, "figures", "fig_framework_overview.png")
FIG_HEADLINE = os.path.join(HERE, "figures", "fig_headline_200.png")
FIG_REACH    = os.path.join(HERE, "figures", "fig_reach_vs_gain.png")

doc = Document()

# ── Style setup ────────────────────────────────────────────────────────────────
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(12)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.15

for level, sz in [(1, 16), (2, 14), (3, 12)]:
    h = doc.styles[f'Heading {level}']
    h.font.name = 'Calibri'
    h.font.size = Pt(sz)
    h.font.bold = True
    h.font.color.rgb = RGBColor(0, 0, 0)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(6)

# Margins
for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

def add_centered(text, size=12, bold=False, italic=False, space_after=6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = 'Calibri'
    return p

def add_para(text, bold=False, size=12, space_after=6, alignment=None):
    p = doc.add_paragraph()
    if alignment:
        p.alignment = alignment
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = 'Calibri'
    return p

def add_mixed(parts, space_after=6, alignment=None):
    """parts = [(text, bold, italic), ...]"""
    p = doc.add_paragraph()
    if alignment:
        p.alignment = alignment
    p.paragraph_format.space_after = Pt(space_after)
    for text, bold, italic in parts:
        r = p.add_run(text)
        r.font.size = Pt(12)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = 'Calibri'
    return p

def add_table(headers, rows, col_widths=None):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(10)
                r.font.name = 'Calibri'
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="D9E2F3"/>')
        cell._tc.get_or_add_tcPr().append(shading)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri+1].cells[ci]
            cell.text = val
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(10)
                    r.font.name = 'Calibri'
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Inches(w)
    return table

def page_break():
    doc.add_page_break()

# ═══════════════════════════════════════════════════════════════════════════════
# FRONT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
doc.add_paragraph()  # spacer
doc.add_paragraph()
add_centered("Task-Aware Active Link Acquisition", size=18, bold=True, space_after=4)
add_centered("for Graph-Based Inference", size=18, bold=True, space_after=24)
add_centered("Project Report Submitted in Partial Fulfilment of the Requirements for the Degree of", size=12, space_after=12)
add_centered("Bachelor of Technology", size=14, bold=True, space_after=4)
add_centered("in", size=12, italic=True, space_after=4)
add_centered("Computer Science and Engineering", size=14, bold=True, space_after=24)
add_centered("Submitted by", size=12, italic=True, space_after=8)
add_centered("Pragunie Aditya  (Roll No. 2310110682)", size=13, bold=True, space_after=24)
add_centered("Under the Supervision of", size=12, italic=True, space_after=8)
add_centered("Dr. Suchi Kumari", size=13, bold=True, space_after=4)
add_centered("Assistant Professor, Department of Computer Science and Engineering", size=11, space_after=4)
add_centered("Shiv Nadar University, Delhi NCR", size=11, space_after=36)
add_centered("Department of Computer Science and Engineering", size=12, bold=True, space_after=4)
add_centered("Shiv Nadar University, Delhi NCR", size=12, bold=True, space_after=4)
add_centered("October, 2025", size=12, space_after=6)

# ═══════════════════════════════════════════════════════════════════════════════
# DECLARATION
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("Declaration", level=1)
add_para(
    "I/We declare that this written submission represents my ideas in my own words and where "
    "others’ ideas or words have been included, I have adequately cited and referenced the "
    "original sources. I also declare that I have adhered to all principles of academic honesty "
    "and integrity and have not misrepresented or fabricated or falsified any idea/data/fact/source "
    "in my submission. I understand that any violation of the above will be cause for disciplinary "
    "action by the University and can also evoke penal action from the sources which have thus not "
    "been properly cited or from whom proper permission has not been taken when needed.",
    space_after=24
)
add_mixed([("Name of the Student: ", True, False), ("Pragunie Aditya", False, False)], space_after=12)

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(4)
r = p.add_run("Signature and Date: ")
r.font.bold = True
r.font.name = 'Calibri'
if os.path.exists(SIG):
    r2 = p.add_run()
    r2.add_picture(SIG, height=Cm(2.0))
r3 = p.add_run("                    14-10-2025")
r3.font.bold = True
r3.font.name = 'Calibri'

doc.add_paragraph()
add_mixed([("Faculty Advisor: ", True, False), ("Dr. Suchi Kumari", False, False)], space_after=12)

p2 = doc.add_paragraph()
p2.paragraph_format.space_after = Pt(4)
r2a = p2.add_run("Signature and Date: ")
r2a.font.bold = True
r2a.font.name = 'Calibri'
if os.path.exists(SIG_SUPERVISOR):
    r2b = p2.add_run()
    r2b.add_picture(SIG_SUPERVISOR, height=Cm(2.0))
else:
    r2b = p2.add_run("_______________________")
    r2b.font.name = 'Calibri'
r2c = p2.add_run("                    14-10-2025")
r2c.font.bold = True
r2c.font.name = 'Calibri'

# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("Abstract", level=1)
add_para(
    "Graph-structured data arising from social, biological, and information networks is frequently "
    "only partially observed, as verifying the existence of a link between two entities may require "
    "expert knowledge, experimental effort, or restricted data access. Given a fixed budget of such "
    "verification queries, this project investigates which candidate links should be prioritised so "
    "that a designated set of target nodes can be classified as accurately as possible. We introduce "
    "a task-aware query scoring framework in which each candidate link is evaluated by its expected "
    "contribution to the classifier’s confidence on the target nodes, weighted by the estimated "
    "probability that the link exists. The classifier employed is label propagation, whose closed-form "
    "solution permits exact evaluation of every candidate without model retraining. On synthetic "
    "community-structured graphs generated under the Contextual Stochastic Block Model, we find that "
    "strategies that maximise the number of true links discovered do not maximise classification "
    "accuracy on the targets. In contrast, the proposed task-aware strategy closes 45–53% of the "
    "accuracy gap between a partially observed and a fully observed network within 200 queries, "
    "compared with at most 13% for standard link-prediction and uncertainty-based heuristics. "
    "These findings replicate on held-out random seeds and remain stable under three distinct "
    "link-hiding schemes. The project further examines the incremental benefit of adaptive "
    "(sequential) query selection and characterises the role of target reachability as the primary "
    "mechanism underlying accuracy improvement. Limitations of the current study and directions for "
    "future work are discussed."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("1. Introduction: Background and Motivation", level=1)

add_para(
    "Relational data from domains such as social networks, citation networks, biological interaction "
    "networks, and knowledge graphs are naturally represented as graphs, where nodes correspond to "
    "entities and edges encode pairwise relationships. A fundamental operation on such graphs is "
    "node classification: assigning each node to one of several discrete categories based on its "
    "position in the network and, optionally, its attributes. In practice, however, the observed "
    "graph is almost never complete. Confirming whether an edge between two entities exists may "
    "require expert annotation, experimental assay, or a paid data access request—each carrying "
    "a non-trivial cost. Consequently, the practitioner faces a budget-constrained setting in which "
    "only a limited number of edge verification queries can be issued."
)

add_para(
    "The problem of deciding which edges to verify has received attention under the framing "
    "of active link prediction, where the goal is to recover as many true links as possible within "
    "the allotted budget. However, recovering more true links does not necessarily translate into "
    "better performance on the downstream inference task. An edge that connects two already-labelled "
    "nodes, or two nodes that are structurally isolated from any labelled node, contributes no "
    "information to a propagation-based classifier regardless of whether it is observed. This "
    "motivates a shift from link-centric to task-centric query selection: choosing edges "
    "not because they are likely to exist, but because discovering them is expected to improve "
    "the classifier’s predictions on a designated set of target nodes."
)

add_para(
    "This project develops and evaluates a task-aware active link acquisition framework. Given a "
    "partially observed graph, a set of labelled nodes, a set of target nodes whose labels are to "
    "be inferred, and a fixed query budget, our method scores each candidate edge by its expected "
    "contribution to classification accuracy on the target nodes. Queries are issued either "
    "adaptively (one at a time, with the scoring updated after each answer) or statically (all "
    "queries planned before the first answer is received). The classifier used throughout is label "
    "propagation, whose closed-form solution enables exact, retraining-free evaluation of every "
    "candidate edge."
)

doc.add_heading("1.1 Research Questions", level=2)
rqs = [
    ("RQ1: ", "Does recovering more true edges lead to higher classification accuracy on the target nodes, or do link-finding rate and task accuracy decouple?"),
    ("RQ2: ", "Does task-aware query selection outperform link-prediction heuristics and generic uncertainty sampling when the evaluation criterion is target accuracy?"),
    ("RQ3: ", "Does adaptive (sequential) query planning yield additional accuracy gains over static planning, and at what point in the budget does this benefit emerge?"),
    ("RQ4: ", "Are the findings robust to different random seeds, different link-hiding schemes, and restriction of the query space to the neighbourhood of the target nodes?"),
]
for label, text in rqs:
    add_mixed([(label, True, False), (text, False, False)], space_after=4)

doc.add_heading("1.2 Objectives", level=2)
objectives = [
    "To construct a controlled experimental benchmark for evaluating edge-query strategies on partially observed graphs with a designated target set and a fixed verification budget.",
    "To demonstrate and quantify the divergence between link-finding performance and downstream classification accuracy on the target nodes.",
    "To design and implement a task-aware value scoring strategy that ranks candidate edges by expected improvement in classifier confidence on the target nodes, and to compare it against structural link-prediction heuristics and generic uncertainty sampling under both static and adaptive planning modes.",
    "To assess the robustness of the proposed strategy through replication on held-out random seeds, evaluation under three distinct link-hiding mechanisms, and restriction of the query space to the frontier of the target nodes’ reachable subgraph.",
]
for i, obj in enumerate(objectives, 1):
    add_para(f"{i}. {obj}", space_after=4)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LITERATURE REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("2. Literature Review", level=1)

add_para(
    "The problem of active link acquisition for downstream graph inference draws on several lines "
    "of prior work. We organise the review from foundational link prediction methods, through "
    "classical machine learning and neural approaches, to active and task-aware learning frameworks."
)

doc.add_heading("2.1 Classical Link Prediction", level=2)
add_para(
    "Link prediction—estimating whether a missing edge exists between two nodes—has been "
    "studied extensively using structural similarity indices derived from the graph topology. "
    "Adamic and Adar [1] proposed a score based on the inverse-log-frequency of common "
    "neighbours, motivated by the intuition that rare common contacts carry more information than "
    "popular ones. Jaccard similarity and the Common Neighbours count are similar "
    "neighbourhood-overlap heuristics. Katz centrality generalises these by summing weighted "
    "paths of all lengths between a pair of nodes. Resource Allocation and the preferential "
    "attachment score (product of degrees) likewise exploit structural properties. These "
    "methods are fast, parameter-free, and interpretable, but they optimise only for edge "
    "existence and carry no information about how a predicted edge would affect a downstream "
    "inference task."
)

doc.add_heading("2.2 Classical Machine Learning Approaches", level=2)
add_para(
    "Beyond purely structural measures, matrix factorisation methods such as SVD-based latent "
    "factor models and Non-negative Matrix Factorisation learn low-dimensional node embeddings "
    "from the adjacency matrix and predict missing entries from their inner product. Logistic "
    "regression and support vector machines trained on hand-crafted node-pair features (degree, "
    "common neighbours, shortest path, Katz score) represent the supervised link prediction "
    "paradigm [12]. These approaches improved over structural heuristics, particularly "
    "on sparse graphs, but they are generally batch methods: a model is trained once on a fixed "
    "observation and then applied, with no mechanism for adaptively choosing which edges to observe."
)

doc.add_heading("2.3 Neural Network and Graph Neural Network Approaches", level=2)
add_para(
    "Graph Autoencoders and Variational Graph Autoencoders [17] introduced end-to-end "
    "learning of node embeddings from both graph structure and node features for link prediction. "
    "The encoder is a Graph Convolutional Network [17] that propagates feature information "
    "over the adjacency matrix; the decoder predicts edge existence from the inner product of "
    "latent representations. Subsequent work replaced the GCN encoder with more expressive "
    "architectures including GraphSAGE, GAT, and SEAL (Weisfeiler-Lehman Subtree-based link "
    "prediction), achieving strong results on standard benchmarks. These methods generalise well "
    "across graphs but remain transductive in the sense that they require the full observed graph "
    "at training time, and they do not address the question of which edges to observe next."
)

doc.add_heading("2.4 Active Link Prediction", level=2)
add_para(
    "The first work to treat the observation process itself as a decision problem is ALPINE by "
    "Chen, Kang, Lijffijt, and De Bie [1], which selects edges to query using a "
    "network-embedding model of link probability and a closed-form V-optimality criterion. "
    "Its simpler selection rules—querying pairs with the highest degree or highest PageRank "
    "score—form two of the baselines used in this project. The key difference is the objective: "
    "ALPINE optimises for link prediction quality (recovering the true adjacency), whereas the "
    "present work optimises for node classification accuracy on a designated target set. "
    "ActiveLink [2] applies deep active learning to link prediction in knowledge "
    "graphs; it likewise targets link recovery rather than a downstream classification task."
)

doc.add_heading("2.5 Active Learning on Graphs and Value-of-Information Scoring", level=2)
add_para(
    "Active learning for node classification has been studied primarily in the label-query setting, "
    "where the budget constrains how many node labels can be obtained. Hu et al. [3] learn a "
    "transferable reinforcement-learning policy, Graph Policy Network, that selects which nodes to "
    "label, with generalisation to unseen graphs. Their setting differs from ours in that "
    "they query labels rather than edges. Bevilacqua et al. [4] learn which subgraphs to "
    "select for subgraph GNNs; while not an active learning method, it supports the principle of "
    "learning a selection policy over graph elements."
)
add_para(
    "The theoretical foundation for value-based query selection is expected error reduction, introduced "
    "by Roy and McCallum [6] for text classification. In this framework, the learner picks the "
    "query whose inclusion is expected to reduce the model’s generalisation error the most. Bayesian "
    "Experimental Design [7] provides a broader framework for optimal data collection under "
    "uncertainty. Li et al. [14] survey deep active learning and organise query strategies into "
    "uncertainty-based, representative, influence-based, Bayesian, and hybrid families; "
    "Ren et al. [13] and Settles [12] provide complementary treatments of the field. "
    "Wilder et al. [15] argue that data acquisition decisions should be evaluated against the "
    "downstream decision they support—the principle motivating our task-aware objective. Foster et "
    "al. [16] study amortised sequential experimental design, a framework relevant to future "
    "work on learning a policy that replaces the value scorer at scale. Cheaper in-the-loop "
    "approximations to expected error reduction include BALD [8, 9], which maximises the mutual "
    "information between model parameters and query outcomes, and influence functions [10], "
    "which estimate per-instance effects via first-order Taylor approximations. Sun et al. [11] "
    "introduce a perturbation-based value signal for graph active learning that is closely related "
    "in spirit to the scoring function developed in this project."
)
add_para(
    "ActiveReach [5] applies active learning to the problem of answering approximate "
    "reachability queries on large graphs. The concept of reachability—whether a target node "
    "lies on a path from a labelled node—also features centrally in our analysis as the primary "
    "mechanism through which edge queries improve classification accuracy."
)

doc.add_heading("2.6 Comparison with the Closest Work", level=2)
add_para("Table 2.1 summarises how the most closely related papers differ from this project.", space_after=4)
add_table(
    ["Work", "Downstream Task", "What is Actively Chosen", "Method"],
    [
        ["ALPINE [1]", "Link prediction", "Edges", "Network embedding + closed-form criterion"],
        ["ActiveLink [2]", "Link prediction in KGs", "Edges", "Deep active learning"],
        ["GPA [3]", "Node classification", "Node labels", "RL policy transferable across graphs"],
        ["PerbALGraph [11]", "Belief representation", "Nodes", "Perturbation-based value signal"],
        ["ActiveReach [5]", "Reachability queries", "Query pairs", "Active learning framework"],
        ["This project", "Node classification on a target set", "Edges",
         "Expected confidence gain under label propagation; adaptive or static scheduling"],
    ],
    col_widths=[1.2, 1.5, 1.2, 2.5]
)
add_para("")  # spacer
add_para("Table 2.1: Comparison of the most closely related prior work with this project.", bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER)

doc.add_heading("2.7 Research Gap", level=2)
add_para(
    "Across the literature surveyed, we did not identify work that (i) selects edges to observe "
    "one at a time, (ii) scores each candidate edge by its expected contribution to node classification "
    "accuracy on a designated target set, and (iii) evaluates this against both structural link-prediction "
    "heuristics and generic uncertainty-based selection in a controlled benchmark with held-out seeds and "
    "multiple link-hiding schemes. The present project addresses this specific combination. We note that "
    "our literature review is not yet a systematic survey; the claim of novelty should be verified against "
    "a broader search before it is asserted in a final submission."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("3. Methodology", level=1)

doc.add_heading("3.1 System Overview", level=2)
add_para(
    "The framework operates as a closed-loop sequential decision process, illustrated in "
    "Figure 3.1. At a high level, it takes as input a partially observed graph, "
    "a small set of labelled nodes, and a designated target set, and it outputs, for each query "
    "step, a decision about which candidate edge to verify next. The central design principle is "
    "that the query decision should be driven by the expected benefit for the downstream inference "
    "task—classifying the target nodes—rather than by the likelihood of finding a true edge."
)

if os.path.exists(FIG_PIPELINE):
    doc.add_picture(FIG_PIPELINE, width=Inches(6.5))
    last_paragraph = doc.paragraphs[-1]
    last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_para("Figure 3.1: EdgeQuery framework overview. The query strategy receives the observed graph, labelled nodes, and target set, selects the next candidate edge to verify, and incorporates oracle answers into the graph. This loop repeats until the query budget is exhausted.", bold=False, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

add_para("The system comprises four interacting components:")
components = [
    ("The environment: ", "a partially observed graph generated under a community structure model, with a fixed fraction of true edges initially revealed and the remainder hidden from the query strategy."),
    ("The oracle: ", "a perfect binary responder that answers each candidate-edge query truthfully. A “yes” answer adds the edge to the observed graph; a “no” removes the pair from the candidate pool."),
    ("The classifier: ", "label propagation applied to the currently observed graph. It propagates label information from known (labelled) nodes to unlabelled nodes along observed edges, computing, for each unlabelled node, a soft belief vector over class labels."),
    ("The query strategy: ", "the component under study. It observes the current graph and the classifier’s beliefs, scores all unqueried candidate edges, and selects one (adaptive mode) or a ranked list (static mode) for the oracle to verify."),
]
for i, (label, desc) in enumerate(components, 1):
    add_mixed([(f"{i}. {label}", True, False), (desc, False, False)], space_after=4)

add_para(
    "The framework is deliberately modular. Any of the four components can be replaced "
    "independently. The oracle could be a noisy responder, the classifier could be a graph neural "
    "network, and the strategy could be a learned policy rather than a hand-designed scorer."
)

doc.add_heading("3.2 Inputs and Outputs", level=2)
add_para("Inputs:", bold=True, space_after=4)
inputs = [
    "G* = (V, E*): the true graph, with |V| = n nodes and |E*| edges. This is not directly observable; only the oracle can access it.",
    "G₀ = (V, E₀): the initially observed graph, where E₀ ⊆ E* is a uniformly sampled fraction ρ of the true edges (ρ = 0.15 in all experiments).",
    "L ⊂ V: the set of labelled nodes, with known class assignments y_L ∈ {1, …, K}.",
    "T ⊂ V \\ L: the set of target nodes, |T| = 15, whose class labels are to be inferred.",
    "B: the query budget, the maximum number of edge queries that may be issued.",
    "C ⊆ (V choose 2) \\ E₀: the set of candidate pairs, all unobserved node pairs.",
]
for inp in inputs:
    add_para(f"•  {inp}", space_after=3)

add_para("Outputs:", bold=True, space_after=4)
outputs = [
    "A sequence of queries (i₁, j₁), (i₂, j₂), …, (i_B, j_B), where each pair is selected from the candidates not yet queried.",
    "For each query step t, an updated observed graph G_t incorporating any confirmed edges.",
    "At step B: the predicted class labels for the target nodes, evaluated by the primary metric:",
]
for out in outputs:
    add_para(f"•  {out}", space_after=3)
add_equation(doc, r"\mathrm{Acc}_T(B) = \frac{1}{|T|} \sum_{u \in T} \mathbf{1}\left[\hat{y}_u = y_u\right]", label="(3.1a)")

doc.add_heading("3.3 Synthetic Network Generation (Contextual Stochastic Block Model)", level=2)
add_para(
    "We generate artificial networks with a standard community model. Each has 90 nodes "
    "in 3 groups. Nodes in the same group are linked with probability 0.13 and nodes in different "
    "groups with probability 0.006, giving about 189 links on average (160–216 across seeds). Each node also has 12 "
    "attributes, but the group only slightly affects them, so the attributes are nearly useless. "
    "This is deliberate: the links must carry the information. A new network is made for each "
    "random seed. Table 3.1 lists all settings."
)
add_para("Table 3.1: Settings used in the experiments.", bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
add_table(
    ["Setting", "Value"],
    [
        ["Nodes, groups", "90 nodes, 3 groups of equal size"],
        ["Link probability within / between groups", "0.13 / 0.006 (about 189 links on average)"],
        ["Node attributes", "12 numbers per node; group signal strength 0.1 (weak by design)"],
        ["Links visible at the start", "Random 15% of the true links"],
        ["Labelled nodes", "15% or 40% of the nodes"],
        ["Target nodes", "15 random unlabelled nodes"],
        ["Query budget B", "200 (400 in one experiment)"],
        ["Seeds", "30 original (0–29) and 30 new (100–129), paired across strategies"],
        ["Classifier", "Label propagation, damping 0.99"],
    ],
    col_widths=[2.5, 4.0]
)

doc.add_heading("3.4 Classifier: Label Propagation", level=2)
add_para(
    "We use label propagation. Each unlabelled node takes on the average belief of its "
    "neighbours, and this repeats until the beliefs settle, with the labelled nodes held fixed. It "
    "is simple and fast to recompute, which lets us test every candidate link exactly."
)
add_para(
    "In symbols, let S = D⁻¹A with D the degree matrix, let U = V \\ L, and let Y_L be "
    "the one-hot labels. With damping α = 0.99 the beliefs of the unlabelled nodes have a "
    "closed form:"
)
add_equation(doc, r"F_U = \left(I - \alpha S_{UU}\right)^{-1} \alpha S_{UL} Y_L, \quad F_L = Y_L", label="(3.1)")
add_para(
    "The predicted class is ŷ_u = argmax_c F_uc, and the class probabilities are "
    "q_u(c) = F_uc / Σ_c' F_uc'. Because this solution is exact, the effect of any one extra "
    "link can be computed without any training."
)

doc.add_heading("3.5 Query Strategies", level=2)
add_para("Table 3.2 lists the strategies compared. Strategies can run static (choose all queries at the start) or adaptive (choose the next query after seeing each answer).", space_after=4)
add_para("Table 3.2: Query strategies compared.", bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
add_table(
    ["Strategy", "Idea in plain words"],
    [
        ["Random", "Pick a pair at random"],
        ["Common neighbours, Adamic–Adar", "Pick pairs that already share friends (likely to be linked)"],
        ["Max-degree, PageRank", "Pick pairs of well-connected nodes (ALPINE-style rules)"],
        ["Uncertainty", "Pick pairs of nodes the classifier is least sure about"],
        ["Value (our method)", "Pretend each link exists, measure how much the classifier’s confidence on the targets improves, and weight by how likely the link is real"],
        ["Value on all nodes", "Same as above, but measuring all unlabelled nodes instead of just the targets"],
    ],
    col_widths=[2.0, 4.5]
)

doc.add_heading("3.6 The Value Strategy in Detail", level=2)
add_para(
    "For a candidate pair (i, j), let P_same(i, j) = Σ_c q_i(c) q_j(c) be the chance that "
    "the two nodes share a class under the current beliefs. A link is more likely between nodes of "
    "the same group, so the estimated chance that the link is real is:"
)
add_equation(doc, r"\hat{P}(ij) = \hat{p}_{in} \cdot P_{same}(i,j) + \hat{p}_{out} \cdot (1 - P_{same}(i,j))", label="(3.2)")
add_para("The classifier’s own estimate of its accuracy on the targets is the average of its largest class probability:")
add_equation(doc, r"\hat{A}_T(A) = \frac{1}{|T|} \sum_{u \in T} \max_c \; q_u(c)", label="(3.3)")
add_para("The score of a pair is the chance that the link is real times the confidence gain if it were:")
add_equation(doc, r"\mathrm{score}(i,j) = \hat{P}(ij) \times \left[\hat{A}_T(A + e_{ij}) - \hat{A}_T(A)\right]", label="(3.4)")
add_para(
    "This is a variant of expected error reduction [6]: it measures the classifier’s own "
    "confidence rather than its error, and it uses no labels beyond the given training labels."
)

doc.add_heading("3.7 Evaluation Protocol", level=2)
add_para("For each run we record:")
add_para("•  Accuracy gain = Acc_T(B) − Acc_T(0)", space_after=3)
add_para("•  Gap closed:", space_after=1)
add_equation(doc, r"\mathrm{gap\;closed} = \frac{\mathrm{Acc}_T(B) - \mathrm{Acc}_T(0)}{\mathrm{Acc}_T^{\mathrm{full}} - \mathrm{Acc}_T(0)}", label="(3.5)")
add_para("•  Reach: the fraction of targets connected by some path to a labelled node", space_after=3)
add_para("•  Number of real links found", space_after=8)
add_para(
    "We used 30 random seeds, and every strategy faced the same networks, so differences are "
    "paired. The p-values quoted in the text are from paired t-tests. For the 12 "
    "headline comparisons we also ran Wilcoxon tests, computed bootstrap "
    "confidence intervals and corrected for running many tests (Holm). The design was tuned on a few "
    "early seeds, so the main result was repeated on 30 new seeds."
)

doc.add_heading("3.8 Implementation", level=2)
add_para(
    "All code is in Python with NumPy and SciPy. Each query needs about "
    "one hundred small linear solves, which is cheap for 90 nodes (tens of seconds per 200 queries "
    "for the adaptive strategy, about a second for the static one)."
)
add_para("Table 3.3: Code organisation.", bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
add_table(
    ["File", "Role"],
    [
        ["src/data.py", "Community-model network generator"],
        ["src/env.py", "Hidden-link environment with the oracle"],
        ["src/labelprop.py", "Iterative label propagation"],
        ["task_aware_al/common.py", "Problem instances, exact LP solver, density estimator"],
        ["task_aware_al/exp_active_2x2.py", "All query strategies and the experiment runner"],
        ["task_aware_al/analysis_*.py", "Cost, frontier, and removal analyses"],
        ["task_aware_al/make_report_figures.py", "Figures for this report"],
    ],
    col_widths=[2.8, 3.7]
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRELIMINARY EXPERIMENTAL EVIDENCE (Initial Results)
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("4. Initial Results", level=1)

add_para(
    "This section summarises the experiments conducted to date and their primary findings. "
    "Detailed quantitative tables and figures will be presented in the final report once "
    "the full experimental suite, including real-world graph benchmarks and a GCN-based classifier, "
    "is complete. A total of approximately 3,000 simulated query runs have been conducted across five experiment "
    "files, covering 17 query strategy variants, two label density settings (15% and 40% of nodes "
    "labelled), budgets of 200 and 400 queries, three link-hiding schemes, and two independent sets "
    "of 30 random seeds each."
)

doc.add_heading("4.1 RQ1 — Link-Finding Rate and Task Accuracy Decouple", level=2)
add_para(
    "The common-neighbours heuristic recovered significantly more true edges than our task-aware method "
    "in 200 queries (10.4 vs 7.5 at 15% labels, p = 0.002), yet our method improved target "
    "classification accuracy far more substantially. The extra edges recovered by common neighbours "
    "raised the fraction of targets connected to a labelled node from only 0.13 to 0.22, whereas "
    "our method raised it to 0.68. This demonstrates that the rate of true-edge discovery is a "
    "poor proxy for downstream classification benefit, because most recovered edges do not lie on "
    "paths connecting target nodes to labelled nodes."
)

doc.add_heading("4.2 RQ2 — Task-Aware Value Scoring Outperforms All Baselines", level=2)
add_para(
    "Across 30 held-out seeds and 200 queries, the adaptive task-aware value strategy achieved an "
    "accuracy gain of +0.209 (15% labels) and +0.238 (40% labels) on the target nodes, closing "
    "45% and 53% of the gap between the partially observed and the fully observed network, "
    "respectively. All heuristic baselines—random, common neighbours, max-degree, PageRank, "
    "Adamic–Adar, and uncertainty sampling—closed at most 13% of this gap. The difference between "
    "the value strategy and each heuristic was statistically significant (p < 0.001, paired t-test; "
    "results replicated with Wilcoxon tests and bootstrap confidence intervals)."
)

if os.path.exists(FIG_HEADLINE):
    doc.add_picture(FIG_HEADLINE, width=Inches(6.2))
    add_para("Figure 4.1: Accuracy gain on target nodes (top) and target reach (bottom) vs. queries spent, "
             "for 15% and 40% labelled nodes. The task-aware value strategy (blue) dominates all heuristic baselines (grey). "
             "Bands show ±1 standard error over 30 seeds.",
             bold=False, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

doc.add_heading("4.3 RQ3 — Adaptive Planning Provides a Modest Incremental Gain", level=2)
add_para(
    "Choosing queries sequentially (re-scoring after each oracle answer) outperformed planning all "
    "queries in advance by approximately +0.05 after 200 queries. This benefit was statistically "
    "significant (p < 0.006) but modest in magnitude and did not emerge reliably before 200 queries. "
    "With a budget of 400 queries, diminishing returns are evident: the marginal gain per 100 queries "
    "for the adaptive value strategy decreases monotonically, from +0.13 in the first 100 to "
    "+0.02 in the final 100, at 15% labels."
)

doc.add_heading("4.4 RQ4 — Robustness Across Seeds and Link-Hiding Schemes", level=2)
add_para(
    "Results replicated closely on a second set of 30 held-out seeds (gains shifted by at most 0.03). "
    "Three link-hiding mechanisms (uniform random, target-biased removal, hub-biased removal) all "
    "produced similar patterns: the task-aware value strategy consistently closed 46–59% of the "
    "gap, while all heuristics remained below 13%. Restricting the query space to the immediate "
    "neighbourhood of target nodes did not reduce the value strategy’s performance."
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. DISCUSSION
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("5. Discussion", level=1)

doc.add_heading("5.1 Why Task Awareness Matters: The Role of Reachability", level=2)
add_para(
    "The central mechanism through which task-aware query selection improves classification can be "
    "understood through the concept of target reachability. In label propagation, the "
    "prediction for a target node can change only if that node lies on a path connecting it to at "
    "least one labelled node in the observed graph. An edge that connects two already-labelled "
    "nodes, or two nodes in a part of the graph disconnected from all labels and targets, conveys "
    "no useful information to the classifier regardless of its existence."
)
add_para(
    "This distinction manifests empirically as a near-perfect correlation between target reach (the "
    "fraction of targets connected to a labelled node after querying) and accuracy gain across the "
    "15 strategy variants examined (r = 0.967 at 15% labels; r = 0.997 at 40% labels). The "
    "dominant driver of the overall correlation is the large gap in reach between "
    "the value strategies and the heuristics: the adaptive value strategy connected 68% of targets "
    "to a label after 200 queries at 15% labels, compared with at most 22% for any heuristic."
)

if os.path.exists(FIG_REACH):
    doc.add_picture(FIG_REACH, width=Inches(6.2))
    add_para("Figure 5.1: Target reach vs. accuracy gain for all strategy variants at B = 200. "
             "The near-perfect correlation (r > 0.96) confirms that connecting targets to labelled nodes "
             "is the primary mechanism behind accuracy improvement.",
             bold=False, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

doc.add_heading("5.2 When Does Sequential Planning Help?", level=2)
add_para(
    "The benefit of adaptive (one-at-a-time) planning over static (all-at-once) planning is "
    "real but modest in the regime studied. A plausible explanation is that the value of information has a "
    "compounding structure: the first confirmed edge on a path to a target enables re-scoring that "
    "identifies the next most valuable edge along that same path. A static planner, which commits "
    "all queries before seeing any answers, cannot exploit this sequential structure."
)

doc.add_heading("5.3 Generalisability of the Framework", level=2)
add_para(
    "Although the experiments presented here evaluate node classification exclusively, the "
    "task-aware active link acquisition framework is not inherently tied to this downstream task. "
    "The pipeline comprises four components—the environment, the oracle, the classifier, and the "
    "query strategy—of which only three pieces depend on the specific inference task: (i) the "
    "task model used as the classifier, (ii) the surrogate quality metric used to score candidate "
    "edges without access to ground-truth labels, and (iii) the final evaluation measure."
)

doc.add_heading("5.4 Limitations of the Current Study", level=2)
limitations = [
    "All networks are synthetic instances of a single model family (cSBM) at one size (90 nodes). The generalisability to real-world graphs has not been established.",
    "The classifier used in the main experiments (label propagation) is the same model the value scorer uses internally, which may give it an inherent advantage.",
    "The ALPINE embedding-based method was not implemented; only its two heuristic selection rules (max-degree and PageRank) were used as baselines.",
    "The separation between task-awareness and candidate-list filtering has not been isolated.",
    "With 15 target nodes, accuracy changes in discrete steps of approximately 0.067; this coarseness limits resolution.",
    "There is no oracle upper bound; the fraction of achievable gain captured is unknown.",
    "Run-time measurements were unreliable across hardware; no quantitative cost comparison is reported.",
]
for lim in limitations:
    add_para(f"•  {lim}", space_after=3)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TIMELINE / PLANNED WORK
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("6. Timeline, Plan After Mid-Term", level=1)

add_para("Table 6.1 lists the planned work in proposed priority order.", space_after=4)
add_para("Table 6.1: Planned future work in proposed priority order.", bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
add_table(
    ["Priority", "Item", "Rationale", "Est. Effort"],
    [
        ["1", "Oracle upper bound and candidate-list ablation",
         "Determines how much of the achievable improvement the value strategy captures", "1–2 weeks"],
        ["2", "Real-world graph benchmarks (Cora, Citeseer, OGBN-arxiv)",
         "Validates generalisability beyond the synthetic cSBM setting", "2–3 weeks"],
        ["3", "GCN as classifier",
         "Tests whether the task-aware principle transfers when retraining-free evaluation requires approximation", "2–3 weeks"],
        ["4", "Full ALPINE method as baseline",
         "Provides a fair comparison with the closest prior work", "1–2 weeks"],
        ["5", "Varied target set definitions",
         "Tests whether findings hold for concentrated or distant targets", "1 week"],
        ["6", "A second downstream task",
         "Demonstrates task-agnosticism; node reachability is the easiest instance", "1 week"],
        ["7", "Learned selection policy",
         "Trains a policy that imitates the value scorer’s decisions for larger graphs", "Stretch goal"],
    ],
    col_widths=[0.6, 1.8, 2.8, 0.9]
)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONCLUSION
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("7. Conclusion", level=1)
add_para(
    "This project investigates a fundamental question in the deployment of graph-based inference "
    "systems: given a partially observed graph and a limited budget for verifying the existence of "
    "edges, which candidate edges should be queried in order to maximise classification accuracy on "
    "a designated set of target nodes? The key finding is that maximising the number of true edges "
    "discovered—the goal of conventional link prediction—is a poor proxy for maximising "
    "downstream accuracy. The task-aware value strategy, which scores each candidate edge by its "
    "expected contribution to the classifier’s confidence on the target nodes, closed 45–53% of "
    "the accuracy gap between the partially observed and the fully observed network within 200 queries, "
    "compared with at most 13% for all heuristic baselines. This advantage was replicated on "
    "held-out random seeds, remained stable under three distinct link-hiding mechanisms, and "
    "persisted when the query space was restricted to the neighbourhood of the target nodes."
)
add_para(
    "The underlying mechanism is target reachability: accuracy improves primarily when queried edges "
    "create paths connecting target nodes to labelled nodes. Adaptive (sequential) query "
    "planning provides an additional but modest gain over static planning, attributable to the "
    "compounding structure of the information value along target-reaching paths."
)
add_para(
    "The current study is limited to synthetic networks under a single community-structure model and "
    "employs label propagation, a classifier whose closed-form solution is integral to the scoring "
    "function. The primary directions for future work are evaluation on real-world citation and "
    "social network benchmarks, integration of a Graph Convolutional Network as the downstream "
    "classifier, and implementation of the full ALPINE method as a stronger baseline."
)

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCES (IEEE format)
# ═══════════════════════════════════════════════════════════════════════════════
page_break()
doc.add_heading("References", level=1)
refs = [
    '[1] Chen, Kang, Lijffijt, De Bie, "Active Link Prediction Using Network Embedding," arXiv:2002.01227, 2020; Applied Sciences 11(11):5043, 2021.',
    '[2] Ostapuk, Yang, Cudré-Mauroux, "Deep Active Learning for Link Prediction in Knowledge Graphs," The Web Conference (WWW), 2019.',
    '[3] Hu et al., "Graph Policy Network for Transferable Active Learning on Graphs," NeurIPS 2020. arXiv:2006.13463.',
    '[4] Bevilacqua, Eliasof, Meirom, Ribeiro, Maron, "Efficient Subgraph GNNs by Learning Effective Selection Policies," ICLR 2024.',
    '[5] Raghebi, Banaei-Kashani, "An active learning framework for approximate reachability query answering in large-scale graphs," Frontiers in Big Data, 2024.',
    '[6] Roy, McCallum, "Toward Optimal Active Learning through Sampling Estimation of Error Reduction," ICML 2001.',
    '[7] Chaloner, Verdinelli, "Bayesian Experimental Design: A Review," Statistical Science, 1995.',
    '[8] Houlsby et al., "Bayesian Active Learning (BALD)," 2011.',
    '[9] Gal, Islam, Ghahramani, "Deep Bayesian Active Learning with Image Data," ICML 2017.',
    '[10] Koh, Liang, "Understanding Black-box Predictions via Influence Functions," ICML 2017.',
    '[11] Sun et al., "Perturbation-based Graph Active Learning for Weakly-Supervised Belief Representation Learning," arXiv:2410.19176, 2024.',
    '[12] Settles, "Active Learning Literature Survey," 2009.',
    '[13] Ren et al., "A Survey of Deep Active Learning," ACM Computing Surveys, 2021. arXiv:2009.00236.',
    '[14] Li et al., "A Survey on Deep Active Learning: Recent Advances and New Frontiers," IEEE TNNLS, 2024. arXiv:2405.00334.',
    '[15] Wilder, Dilkina, Tambe, "Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization," AAAI 2019.',
    '[16] Foster, Ivanova, Malik, Rainforth, "Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design," ICML 2021.',
    '[17] Kipf, Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017.',
    '[18] Deshpande et al., "Contextual Stochastic Block Models," NeurIPS 2018.',
    '[19] Zhu, Ghahramani, Lafferty, "Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions," ICML 2003.',
    '[20] Adamic, Adar, "Friends and Neighbors on the Web," Social Networks, 2003.',
]
for ref in refs:
    add_para(ref, size=11, space_after=4)

# ═══════════════════════════════════════════════════════════════════════════════
doc.save(OUT)
print("Saved:", OUT)
