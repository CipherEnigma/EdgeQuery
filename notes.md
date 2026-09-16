## initial
We are implementing a GCN to demonstrate it can classify nodes correctly using graph structure, even when node features are useless.

# E0 experimental-->pipeline 17 sept
## Part 1: What problem are we even solving?
Imagine you have a group of people, and each person belongs to a secret "team" (Team A, Team B, or Team C) — that's their class. You don't get told directly which team someone is on. Instead, you get two clues:

Who talks to whom (a social network / graph) — people tend to talk more to others on their own team.
A little bit of info about each person (their features) — like a short bio, but a really vague/noisy one that barely hints at their team.
Your job: build a machine that looks at both clues and guesses everyone's team correctly.

Why this is interesting: if the "bio" clue is too strong on its own, you don't even need the "who talks to whom" clue — a simple model could just read bios and be done. So to make this a genuinely interesting test of "can a model use network structure," we deliberately make the bios weak (barely informative) and make the "who talks to whom" pattern strong (clearly cliquish by team). That way, doing well forces the model to actually exploit the network structure — which is the whole point of using a GCN (Graph Convolutional Network), a type of model built specifically to learn from graphs


## Part 2: Deciding who's on which team (y)
We have 90 people (nodes), and we want to split them evenly into 3 teams (classes 0, 1, 2) — 30 people per team.

The simplest way to think about it: line up all 90 people in a row, and just label the first 30 as "Team 0," the next 30 as "Team 1," and the last 30 as "Team 2." There's no cleverness here — the actual order of people doesn't matter, because nothing about "who is person #5" is meaningful yet; it's just an ID number.

The result is stored in something called y — a list of 90 numbers, where y[i] tells you person i's team. So it'd look like:
y = [0, 0, 0, ..., 0,   1, 1, 1, ..., 1,   2, 2, 2, ..., 2]
     └── 30 zeros ──┘   └── 30 ones ──┘   └── 30 twos------



## Part 3: Deciding who talks to whom (A, the edges)
Now that everyone has a team, we need to decide the actual connections. The rule is simple: same-team pairs are much more likely to be connected than different-team pairs.

How the code does it, conceptually
Think of every possible pair of people — person 1 & person 2, person 1 & person 3, all the way through every possible pairing among the 90 people. For each pair, we do something like flipping a biased coin:

If both people are on the same team → flip a coin that lands "connected" 11% of the time (p_in = 0.11).
If they're on different teams → flip a coin that lands "connected" only 0.5% of the time (p_out = 0.005).
That's it — go through every pair once, flip the appropriate coin, and if it lands "connected," draw that edge.

Because p_in (11%) is so much bigger than p_out (0.5%) — about 22x bigger — the end result is a graph where people cluster tightly with their own team and only rarely cross-connect. That's the "structure carries the signal" property we want.

### Why we only check each pair once
We only ever check each pair a single time (person 3 & person 47 — not separately checking "47 & 3" again), and when we do connect them, we mark both directions in the grid at once (A[i,j] and A[j,i] both become 1). This guarantees the matrix stays symmetric by construction — there's no way for it to accidentally end up lopsided.

End result: a 90×90 grid of 0s and 1s, symmetric, zero diagonal, where same-team connections vastly outnumber different-team ones. When we ran this, we got 148 total edges — 139 same-team, only 9 different-team, roughly a 15:1 ratio.


## Part 4: Building each person's "bio" (X, the features)
Now we give every person a short numeric profile — 12 numbers per person (feat_dim = 12). Think of it like 12 vague personality-test scores. The trick is: these numbers should hint at someone's team, but only very faintly — mostly they should just look like random noise.

How it's built, conceptually
Step A — Give each team a "typical profile."
Before looking at individual people at all, we invent one random 12-number profile per team — call these the team's center. So Team 0 gets some random profile like [0.4, -1.2, 0.8, ...], Team 1 gets a different random profile, Team 2 gets another. These are just fixed reference points, picked once, randomly.

Step B — Give each person their own bio, nudged slightly toward their team's center.
For each of the 90 people:

Roll up 12 completely random numbers (pure noise, nothing to do with their team) — this is their "raw" bio.
Add a tiny nudge toward their team's center profile from Step A.
So: person's bio = random noise + a small pull toward their team's typical profile.

### what is feat signal 
feat_signal (0.10) — this controls how hard we pull each person's bio toward their team's center. Think of it as a dial: at 0, there's no pull at all (pure random noise, features tell you nothing about team). At 0.10, the pull is there but small — like a whisper compared to the noise, which is comparatively loud. If we cranked this dial up to, say, 2.0, everyone's bio would end up clustered tightly near their team's center, and you could guess someone's team just from their bio alone — no graph needed. We deliberately keep it low so that doesn't happen.


### Why do this at all instead of pure noise?
We talked about this before — real-world data is almost never pure noise; there's usually a faint real signal in features, just not enough to be decisive by itself. This mimics that realistically, and gives us a tunable "signal strength" if we ever want to experiment with stronger or weaker bios later.

End result: a 90×12 grid of numbers (X) — mostly noise, each row (each person) faintly resembling their team's center profile, but not enough to reliably tell teams apart just by looking at the bios.


## Part 5: What is a GCN even trying to do?
Before we get into the code, let's build the mental picture.

Imagine you're trying to guess someone's team, and you're given their (weak, noisy) bio. That alone won't get you very far. But now imagine you're also told: "here are the 5 people this person talks to most." Even without knowing anything else about them, you could reasonably guess: "well, if 4 out of 5 of their friends are clearly Team A, this person's probably Team A too" — because we know from Part 3 that same-team people cluster together.

That's the entire idea behind a GCN (Graph Convolutional Network): instead of judging a person purely on their own weak bio, let them "absorb" information from their neighbors' bios too. Do that a couple of times (spread information a couple of hops through the network), and suddenly the noisy individual bios start averaging out into something much more useful — because your teammates' noise doesn't systematically point the same wrong direction as your noise, but the very faint "pull toward team center" signal does line up across teammates and reinforces.


In short: a GCN layer does two things —

Transform: apply some learnable math to every node's own features (a normal neural-network step).
Aggregate: replace each node's result with a (weighted) average of what its neighbors computed in step 1.
Stack two of these, and a node's final representation reflects not just itself, but its neighbors, and its neighbors' neighbors (2 hops out) — which is exactly the "ask your friends" intuition from the top of this section


## Part 6+7: Recap of normalize_adj, then the GCNLayer and GCN classes
Quick recap of Part 6: normalize_adj takes the raw 0/1 adjacency matrix A and turns it into A_norm — a rescaled version that (1) adds self-loops so a node includes its own data, and (2) balances out "popular" vs. "unpopular" nodes so the later averaging step is fair. A_norm is what actually gets fed into the model.

The GCNLayer — one step of "transform, then average with neighbors"

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A_norm):
        return A_norm @ self.linear(X)
Two things happen here, in order:

### self.linear(X) — 
apply a learnable transformation to every node's own features independently first, with no neighbor info involved yet. Think of nn.Linear as: take each person's 12-number bio, multiply it by some adjustable weight-numbers and add some adjustable bias-numbers, producing a new set of numbers. Every node goes through the exact same transformation (the same weights), just applied to their own individual data.

### A_norm @ (...) — 
now spread that transformed data across the graph using matrix multiplication with A_norm. This is the "averaging with neighbors" step from Part 5's intuition — each node's new value becomes a weighted combination of its own (already-transformed) data and its neighbors' (already-transformed) data.


The GCN — stacking two layers together

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, X, A_norm):
        h = self.layer1(X, A_norm)
        h = F.relu(h)
        out = self.layer2(h, A_norm)
        return out
layer1: takes each node's raw 12-number bio → produces a "hidden" representation (we chose 16 numbers) that already reflects 1 hop of neighbor-averaging.
F.relu(h): a simple rule applied to every number: "if it's negative, make it 0; if it's positive, leave it alone." This is called an activation function, and it's what lets stacking multiple layers actually add expressive power — without it, two linear layers back-to-back are mathematically equivalent to just one bigger linear layer, so you'd gain nothing from stacking. ReLU (Rectified Linear Unit) is just the simplest, most common choice.


layer2: takes that hidden representation (which already reflects 1-hop neighbors) and does the transform+aggregate step again → now each node's output reflects a 2-hop neighborhood (its neighbors' neighbors too). The output size here is num_classes (3) — one score per possible team.
out: raw per-team scores (called logits) — not yet "probabilities," just relative scores where a higher number means "more likely this team." Converting these into clean probabilities and an actual training signal happens in the next part.
That's the whole model architecture: raw noisy bio in → two rounds of "combine with neighbors" → 3 scores out, one per team, per node.


## Part 8: Training, accuracy, and uncertainty
Splitting nodes into train vs. test

def split_nodes(n, train_frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return ...

Before training, we need to hide some people's team labels from the model, so we can later check "did it actually learn to generalize, or did it just memorize?" rng.permutation(n) shuffles the 90 node indices into random order, then we cut that shuffled list in half: the first half becomes train nodes (labels the model is allowed to learn from), the second half becomes test nodes (labels we hide during training, and only check afterward to grade the model).

Important nuance: all 90 people and all their connections stay visible in the graph the whole time — the model can still see a test person's bio and who they talk to. We only hide their team label during training. This matters because a GCN needs the full graph structure to do its neighbor-averaging trick properly, even for test nodes.

Train / test split — a standard ML practice: never let the model be graded on the exact same data it learned from, or you can't tell if it actually generalized vs. just memorized. train = "learn from this," test = "grade yourself on this, but only look at it at the end."
Permutation — a random shuffle/reordering of a list.
Training the model

def train_model(model, X, A_norm, y, train_idx, epochs=200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X, A_norm)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
    return model
Think of training as: "run the model, see how wrong it was, nudge its internal numbers slightly to be less wrong, repeat many times."

model(X, A_norm) — run the forward pass (everything from Part 6+7): produces 3 scores per node.
out[train_idx], y[train_idx] — only look at the train nodes' scores and true labels — we deliberately never let the loss "see" test-node labels.
F.cross_entropy(...) — this is the loss function: a single number measuring "how wrong were the model's predictions, overall." It compares the model's scores against the true labels and produces a bigger number when predictions are more confidently wrong, smaller when they're right. This is the number we're trying to shrink.
loss.backward() — this is backpropagation: PyTorch automatically works out, for every single learnable number in the model, "if I nudged this number up or down slightly, would the loss go up or down, and by how much?" This is called computing the gradient.
optimizer.step() — actually apply those nudges. Adam is a specific, well-tested strategy for deciding exactly how big each nudge should be (it's smarter than just "always nudge by a fixed tiny amount" — it adapts).
optimizer.zero_grad() — housekeeping: clear out the previous round's nudge-calculations before computing fresh ones, since otherwise they'd accidentally pile up across rounds.
epochs=200 — how many times we repeat this whole "predict → measure wrongness → nudge" cycle. Each full pass through this cycle is called an epoch. More epochs generally means more learning, up to a point.
lr=0.01 (learning rate) — how big each nudge is allowed to be. Too big and the model overshoots and never settles; too small and training is painfully slow. 0.01 is a common reasonable default.
Checking accuracy

def accuracy(model, X, A_norm, y, idx):
    model.eval()
    with torch.no_grad():
        out = model(X, A_norm)
        preds = out[idx].argmax(dim=1)
        return (preds == y[idx]).float().mean().item()
model.eval() — tell PyTorch "we're just checking, not training" (some behaviors differ slightly between training-mode and evaluating-mode, though our simple model doesn't actually have any of those differences — it's just good practice).
torch.no_grad() — tells PyTorch "don't bother tracking how to compute nudges this time," since we're not training right now — this just makes it run faster.
argmax(dim=1) — out of each node's 3 scores, pick whichever team got the highest score — that's the model's actual guess.
(preds == y[idx]).float().mean() — compare guesses to true labels, count what fraction were correct. That fraction is the accuracy (e.g. 0.711 = 71.1% correct).
Measuring uncertainty

def entropy_uncertainty(model, X, A_norm):
    ...
    probs = F.softmax(out, dim=1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=1)
F.softmax(out, dim=1) — converts the 3 raw scores per node into 3 numbers that add up to 1 and are all positive — i.e., actual probabilities ("I'm 70% sure Team 0, 20% sure Team 1, 10% sure Team 2").
Entropy — a standard math measure of "how spread out / uncertain" a probability distribution is. If the model says "99% Team 0, 0.5% Team 1, 0.5% Team 2," that's low entropy (very confident). If it says "34% / 33% / 33%," that's high entropy (basically clueless — as uncertain as a random guess). This function computes exactly that per node, and we'll reuse it later in Step 5 as a way to pick "which unknown edge should we investigate" — intuitively, prioritize learning more about nodes the model is currently most unsure about.


The headroom check — why this all matters
Running model.py directly trains two versions: one using the real graph, one using no real edges at all (features only). Result: 71.1% accuracy with the graph vs. 31.1% without it — a huge 40-point gap. That's the proof that this whole setup is doing its job: the graph structure is genuinely necessary, features alone are nowhere near enough.

## Part 9: env.py and loop.py — the "pay to learn more edges" system
The idea, in plain terms
So far, we've assumed the model gets to see the whole graph at once. But imagine a more realistic situation: discovering a connection between two people is expensive — maybe it requires a survey, an investigation, whatever. You don't get the whole graph for free. Instead, you start with just a tiny peek at it, and you have a limited budget to "ask" about specific pairs of people: "are these two connected?" Each time you ask, you spend one unit of budget and get a truthful yes/no answer.

The research question this whole project is really about: if you only get to ask a limited number of these questions, which pairs should you ask about first, to improve your model's accuracy as much as possible? Randomly? Or is there a smarter strategy?

This is a well-known idea in ML called active learning — instead of passively receiving data, you get to strategically choose what to learn about next.

env.py — the world that hides edges and answers questions
GraphEnv wraps the real graph from data.py and deliberately hides most of it:

A_true — the full real graph (ground truth), known internally, never shown directly.
A_obs ("observed") — what the model is actually allowed to train on. Starts with only 10% of the real edges visible.
train_idx / val_idx / test_idx — same idea as before, but now split three ways: train (labels to learn from), val (a "practice test," typically used later for tuning decisions), test (the untouched final judge — never used for anything except the final accuracy number).
candidates() — the list of every pair of people we haven't asked about yet. Some of these pairs are secretly real edges, some aren't — we genuinely don't know until we ask.

reveal(edge) — the oracle: you hand it a pair, and it truthfully tells you "yes, that's a real connection" or "no, it isn't" — and if yes, it actually adds that edge into A_obs so the model can now see it. Either way, that pair is now "used up" — you can't ask about it again (that would be like a free retry, unrealistic and unfair).

Jargon, defined plainly:
Budget — how many questions ("reveals") you're allowed to ask in total before you have to stop and just use whatever graph you've built up.
Oracle — a term borrowed from ML/theory meaning "a source that truthfully answers a specific question when asked" — here, "is this edge real?"
Candidate — an unasked pair; a possible thing you could spend budget on.
Leakage — when information "leaks" somewhere it shouldn't — e.g. if hidden edges accidentally showed up in A_obs before being revealed, or if test-node labels somehow influenced training. We wrote 5 explicit checks to make sure none of this can happen (all passing) — this matters because if leakage happened, our accuracy numbers would be fake/inflated and the whole experiment untrustworthy.
loop.py — the engine that actually spends the budget
run_loop(env, scorer_fn, budget, ...) repeats this cycle:

Train a fresh GCN on whatever's currently in A_obs.
Record how accurate it is on the test nodes.
Use a scorer function to rate every remaining candidate pair — "how promising is asking about this one?"
Reveal (ask about) whichever candidate scored highest.
Repeat, until the budget runs out.
At the end, you get an accuracy curve — how test accuracy improved (or didn't) as budget got spent.

Scorer — a swappable strategy for ranking candidates. Right now we only have random_scorer (just picks randomly — the simplest possible baseline, used to sanity-check the machinery works at all). Later steps will add smarter scorers: e.g. "ask about pairs involving nodes the model is most uncertain about" (using entropy_uncertainty from model.py).
What we found when we ran it
With random scoring, most of the budget gets wasted: because real edges are rare (~148 real ones out of ~4,000 possible pairs, roughly 3.7%), randomly guessing pairs mostly lands on "no, not real" — so the observed graph barely grows, and accuracy barely moves. With a 10-question budget, accuracy stayed completely flat. With 100 questions, it very slowly crept from 22% up to 33% as a few lucky real edges got found.

This isn't a failure — it's actually the expected, important result: it sets up the real test later (Step 8, the "kill test") — a genuinely smart scoring strategy should reach much higher accuracy using far fewer questions than random needs, by being deliberate about which pairs are worth asking about. If it can't beat random, that tells us the "smart strategy" idea doesn't actually work here.

That's the complete picture of everything built so far:


data.py   → builds the fake world (teams, weak bios, true connections)
model.py  → the GCN that learns to guess teams from bios + graph structure
env.py    → hides most connections, answers yes/no questions about them, tracks a budget
loop.py   → spends the budget by repeatedly training, scoring candidates, and asking the best one


## Part 10: scorers.py — the actual strategies being tested

random_scorer was always just a baseline to prove the machinery works. Now we build the strategies we actually care about comparing. They all share one shape: scorer_fn(env, candidates, model, X, A_norm) -> an array of numbers, one score per candidate. The loop always just picks whichever candidate scored highest. Because every scorer shares this exact interface, they're fully swappable — the loop doesn't need to know or care which strategy it's running.

### uncertainty_scorer
Uses entropy_uncertainty (built back in Part 8) to find, for every node, how unsure the model currently is about its team. For a candidate pair (i, j), the score is just entropy[i] + entropy[j] — "how confused is the model about both endpoints of this potential edge, combined." Intuition: if the model is already confident about a node, learning more about its connections probably won't change much; if it's confused, a new edge might resolve that confusion.

### structural_scorer
Doesn't touch the model at all — purely looks at the currently observed graph. For a candidate (i, j), it counts shared neighbours: how many other nodes are already connected to both i and j in A_obs. Intuition (this is a very old idea in network science, sometimes called "common neighbours"): if two people already share a lot of mutual connections, they're statistically more likely to be connected themselves too — same-team clusters tend to be densely interconnected, so shared neighbours is a cheap proxy for "these two are probably on the same team, and probably connected."

### oracle_voi_scorer (Value of Information)
The expensive, "just try it and see" strategy. For a sample of candidates (not all of them — too slow), it:
1. Tentatively adds that one edge to a temporary copy of the observed graph (regardless of whether it's actually real — this is a hypothetical "what if" test).
2. Retrains a fresh GCN from scratch on that temporary graph.
3. Checks accuracy on the validation set (never the test set — using test accuracy here would leak test information into the edge-selection decision itself).
4. Scores that candidate by the resulting validation accuracy.

This is the "gold standard" in principle, because it directly measures what we actually care about (does this edge help the model) instead of using an indirect proxy like entropy or shared neighbours. The catch: it's extremely expensive (a full retrain per candidate), so in practice we only sample a small number of candidates (n_sample=15) and train each with a reduced number of epochs (25) to keep runtime reasonable. Un-sampled candidates get a score of -infinity so they're never accidentally picked over evaluated ones.

### Jargon, defined plainly:
- Value of Information (VOI) — a decision-theory idea: instead of guessing which action might be useful, you directly (even if expensively) measure how useful each option actually is before committing.
- Validation set — a third slice of data (separate from train and test) used to make decisions during the process (like "which edge should I pick") without touching the test set, which stays reserved purely for the final grade.


## Part 11: eval.py and multi-seed runs — turning curves into a verdict

One run of the loop gives you one accuracy curve (accuracy at each budget step). But a single run could just be lucky or unlucky, depending on random initialization, which edges happened to be pre-observed, etc. So before trusting any comparison between strategies, we needed two more things.

### run_multi_seed (added to loop.py)
Runs the entire loop multiple times with different random seeds (same underlying graph, different environment/model randomness each time), and stacks all the resulting accuracy curves together. This is what lets us report an average with a sense of how much the result varies run-to-run, instead of trusting one potentially-lucky run.

### auc() — turning a curve into one number
AUC = "area under the curve." Instead of comparing two long lists of numbers (accuracy at every single step) between strategies, we collapse each curve into a single summary number: roughly, "how good was this strategy, averaged across the whole budget-spending process," not just at the very end. A strategy that gets to high accuracy quickly and stays there scores a higher AUC than one that only gets there at the very last step, even if their final accuracy is identical — AUC rewards being good early, not just eventually.

### mean_and_se() and plot_curves()
Once we have curves from multiple seeds, mean_and_se computes the average curve plus the standard error (a measure of how much the average could be off, given how much the seeds disagreed with each other). plot_curves draws all strategies on one graph, with a shaded band around each line showing that uncertainty — so you can visually tell whether two strategies' curves are meaningfully different or just noisy overlap.


## Part 12: E0 and E1 — running the actual experiments

### experiments/e0_sanity.py — "is the harness honest?"
Runs random_scorer vs uncertainty_scorer, 5 seeds each, budget=40. This is not meant to prove one strategy beats another — it exists purely to catch bugs (leakage, crashes, a scorer that makes things obviously worse). It passes as long as uncertainty isn't dramatically worse than random.

Result: random AUC 12.14, uncertainty AUC 11.83 — roughly a tie. Sanity check passed: the pipeline behaves honestly.

### experiments/e1_killtest.py — "does being smart actually help?"
Runs all four strategies (random, structural, uncertainty, oracle_voi), 3 seeds each, budget=20, and reports the AUC gap of each strategy vs random.

Result:
- structural: +0.67 AUC vs random -> the only strategy with a clear positive gap
- uncertainty: -0.11 AUC vs random -> essentially flat, no benefit
- oracle_voi: +0.06 AUC vs random -> barely moved, likely because it was starved of compute (only 15 sampled candidates, 25 training epochs per probe, ~18-node validation set)

Verdict from this run: structural -> PURSUE.

### Why we ran E0 and E1 as two separate things
E0 answers "is the code trustworthy?" — a check on the implementation. E1 answers "is the idea any good?" — a check on the research question itself. Running E0 first means that if E1 comes back with a weak or null result, we can trust that it's telling us something real about the idea, not just revealing a hidden bug. Skipping E0 would leave that ambiguous.

### Why "pursue structural" specifically
It's the only strategy that showed a clear, positive AUC gap over random. "Pursue" doesn't mean "this is definitely the final answer" — it means "this is the one signal worth spending more experimental effort confirming," rather than abandoning the whole active-edge-querying idea.

### Should this be alarming / cause a change of direction?
No, but there are real caveats to hold onto:
- The AUC gaps (especially +0.67 for structural) are similar in size to the seed-to-seed standard deviation (~0.9-1.0), so with only 3 seeds this isn't yet a confident result — more seeds are needed before trusting it fully.
- oracle_voi underperforming is more likely a compute-budget artifact (too few sampled candidates, too few training epochs, tiny validation set) than real evidence against the "directly measure the benefit" idea.
- uncertainty being flat is a more genuine negative result worth keeping in mind — a cheap heuristic (structural) currently beats a more "principled-sounding" one (uncertainty).

Net: this says "tighten the experiment and gather more evidence," not "the idea is broken, change direction."

### Questions worth bringing to the professor before moving forward
- How many seeds would be considered enough to trust an AUC gap like structural's +0.67, given the observed seed-to-seed standard deviation? Should a significance test (e.g. paired t-test across seeds) be used instead of comparing raw means?
- Oracle-VOI is supposed to be the theoretical best case but underperformed a cheap heuristic here, likely due to a heavy approximation (small candidate sample, few training epochs, small validation set) needed to keep runtime reasonable — is there a cheaper-but-more-faithful way to estimate it, or is this tradeoff acceptable to report as a known limitation?
- Is keeping the graph small (~90 nodes, so a single edge can actually move the score) a reasonable scale choice, or should these findings also be tested at larger scale even if the per-edge effect gets diluted?
- Are observe_frac=0.1 and budget=20 reasonable fixed defaults, or should they be swept as part of the experiment rather than fixed?
- Is a real-but-modest effect size (like structural's) on a synthetic benchmark presentation-worthy as-is, or does it need a stronger effect size / a second benchmark first?
- If structural keeps winning with more seeds, is the natural next experiment to vary homophily strength (p_in / p_out) and see whether structural's advantage grows or shrinks as the graph's community structure gets stronger or weaker — i.e., investigating *why* it wins, not just *that* it wins?












