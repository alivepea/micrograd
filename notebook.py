import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import random

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_moons

    from micrograd.engine import Value
    from micrograd.nn import MLP
    from micrograd.viz.model_graph import ModelGraph, trace_sample, view_height

    return (
        MLP,
        ModelGraph,
        Value,
        make_moons,
        mo,
        np,
        plt,
        random,
        trace_sample,
        view_height,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MicroGrad demo

    用一个很小的多层感知机，在二维双月数据上做二分类。前向、损失和更新都走 `micrograd`：标量自动求导，上面是一个 PyTorch 风格的 `MLP`。

    样本数、噪声和网络宽度会立刻反映到图上。SGD 一次要跑一会儿，改完参数后点 **训练**。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 数据

    `make_moons` 造出两弯交错的点。标签从 $\{0, 1\}$ 改成 $\{-1, +1\}$，后面的间隔损失就可以直接乘上 `yi * score`。
    """)
    return


@app.cell
def _(mo):
    n_samples = mo.ui.slider(
        40, 160, step=10, value=100, debounce=True, show_value=True, label="样本数"
    )
    noise = mo.ui.slider(
        0, 0.35, step=0.01, value=0.1, debounce=True, show_value=True, label="噪声"
    )
    seed = mo.ui.number(0, 9999, step=1, value=1337, debounce=True, label="随机种子")
    mo.hstack([n_samples, noise, seed], justify="start", align="end", gap=1, wrap=True)
    return n_samples, noise, seed


@app.cell
def _(make_moons, n_samples, noise, seed):
    X, y = make_moons(
        n_samples=int(n_samples.value),
        noise=float(noise.value),
        random_state=int(seed.value),
    )
    # Map {0, 1} onto {-1, +1} for the max-margin loss.
    y = y * 2 - 1
    return X, y


@app.cell
def _(X, mo, plt, y):
    data_fig, data_ax = plt.subplots(figsize=(5, 5))
    data_ax.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
    data_ax.set_aspect("equal", adjustable="box")
    data_ax.set_title("moons")
    data_fig.tight_layout()
    mo.vstack(
        [
            data_fig,
            mo.md(f"{len(y)} 个点，正类 {(y > 0).sum()}，负类 {(y < 0).sum()}。"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 模型

    两个等宽的 ReLU 隐层，最后一层线性输出一个分数。默认 `2 → 16 → 16 → 1`。输出层是 16 个权重加 1 个偏置，所以有 17 个参数；整个网络默认 337 个标量参数。

    下面的损失和准确率是**训练前**的。同一颗种子会得到同一组初始权重。
    """)
    return


@app.cell
def _(mo):
    hidden = mo.ui.slider(4, 32, step=4, value=16, debounce=True, show_value=True, label="隐层宽度")
    steps = mo.ui.slider(
        20, 120, step=10, value=100, debounce=True, show_value=True, label="训练步数"
    )
    train_button = mo.ui.run_button(label="训练", kind="success")
    mo.hstack([hidden, steps, train_button], justify="start", align="end", gap=1, wrap=True)
    return hidden, steps, train_button


@app.cell
def _(MLP, Value, X, hidden, np, random, seed, y):
    def loss(model, features, labels, batch_size=None):
        # Inline loader. The demo trains on the full set.
        if batch_size is None:
            Xb, yb = features, labels
        else:
            ri = np.random.permutation(features.shape[0])[:batch_size]
            Xb, yb = features[ri], labels[ri]
        inputs = [list(map(Value, xrow)) for xrow in Xb]

        # Forward the model to get scores.
        scores = list(map(model, inputs))

        # SVM max-margin (hinge) loss.
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        # L2 regularization.
        alpha = 1e-4
        reg_loss = alpha * sum(p * p for p in model.parameters())
        total_loss = data_loss + reg_loss

        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)

    def fresh_model():
        random.seed(int(seed.value))
        width = int(hidden.value)
        return MLP(2, [width, width, 1])

    def layer_rows(model):
        # Per-layer parameter counts, including the output layer's 17.
        lines = [
            "| 层 | 非线性 | 输入 | 神经元 | 参数 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for index, layer in enumerate(model.layers):
            neuron = layer.neurons[0]
            kind = "ReLU" if neuron.nonlin else "线性"
            lines.append(
                f"| {index} | {kind} | {len(neuron.w)} | {len(layer.neurons)} | {len(layer.parameters())} |"
            )
        return "\n".join(lines)

    preview = fresh_model()
    initial_loss, initial_acc = loss(preview, X, y)
    return fresh_model, initial_acc, initial_loss, layer_rows, loss, preview


@app.cell
def _(initial_acc, initial_loss, layer_rows, mo, preview):
    width = len(preview.layers[0].neurons)
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"2 → {width} → {width} → 1", label="结构", bordered=True),
                    mo.stat(str(len(preview.parameters())), label="参数量", bordered=True),
                    mo.stat(
                        f"{initial_loss.data:.3f}",
                        label="训练前损失",
                        bordered=True,
                        target_direction="decrease",
                    ),
                    mo.stat(
                        f"{initial_acc * 100:.0f}%",
                        label="训练前准确率",
                        bordered=True,
                    ),
                ],
                justify="start",
                gap=1,
                wrap=True,
            ),
            mo.md(layer_rows(preview)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 一张样本上的计算图

    从左到右是输入、两层 ReLU、线性分数，最右边是这条样本的 hinge：$\mathrm{ReLU}(1 - y \cdot \mathrm{score})$。边上是权重，蓝正红负，越粗 $|w|$ 越大。节点里的数是前向值，边框颜色是梯度的符号。

    点一个神经元，下面列出 $b + \sum_i w_i x_i$ 的每一项，以及这项对**这条样本的 hinge** 的梯度。梯度不是训练步里那个全量损失的梯度。拖拽平移，按钮缩放。
    """)
    return


@app.cell
def _(mo, y):
    sample_index = mo.ui.slider(
        0,
        len(y) - 1,
        step=1,
        value=0,
        debounce=True,
        show_value=True,
        label="样本",
    )
    sample_index
    return (sample_index,)


@app.cell
def _(ModelGraph, X, mo, preview, sample_index, trace_sample, view_height, y):
    index = int(sample_index.value)
    elements, summary = trace_sample(preview, X[index], float(y[index]))
    model_graph = mo.ui.anywidget(
        ModelGraph(elements=elements, summary=summary, height=view_height(preview))
    )
    model_graph
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 同一条样本的反向

    布局和上面一样，读法反过来。节点里的数是梯度。箭头从 hinge 指回输入：边上的粗细是 $|\partial w|$，绿是正、橙是负、灰是 0。ReLU 预激活 $\le 0$ 时，整条支路的梯度都是 0。

    点一条边看 $\partial w = x \cdot \partial(w \cdot x)$。这仍然只是这一条样本的 hinge，不是训练时把所有样本加起来的梯度。
    """)
    return


@app.cell
def _(ModelGraph, X, mo, preview, sample_index, trace_sample, view_height, y):
    back_index = int(sample_index.value)
    back_elements, back_summary = trace_sample(
        preview, X[back_index], float(y[back_index]), view="backward"
    )
    backward_graph = mo.ui.anywidget(
        ModelGraph(
            elements=back_elements,
            summary=back_summary,
            height=view_height(preview),
            mode="backward",
        )
    )
    backward_graph
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 训练

    每一步都是全量前向、反向，然后 SGD。学习率从 $1$ 线性收到大约 $0.1$：

    $$
    \eta_k = 1 - 0.9 \cdot k / T
    $$

    $T = 100$ 时，这就是原来 demo 里的 `1 - 0.9 * k / 100`。损失是 hinge，再加 `1e-4` 的 L2。
    """)
    return


@app.cell
def _(Value, X, fresh_model, loss, mo, np, plt, steps, train_button, y):
    # Script runs (uv run notebook.py) train immediately. In the editor, wait
    # for the button so slider changes do not restart an ~80s loop.
    mo.stop(
        mo.app_meta().mode != "script" and not train_button.value,
        mo.callout(
            "点「训练」跑 SGD。数据和宽度可以先改，曲线和决策边界会在这次训练之后更新。",
            kind="info",
        ),
    )

    model = fresh_model()
    n_steps = int(steps.value)
    history = []
    for k in mo.status.progress_bar(range(n_steps), title="SGD", remove_on_exit=True):
        total_loss, acc = loss(model, X, y)
        model.zero_grad()
        total_loss.backward()
        # Linear decay from 1.0 toward 0.1. At 100 steps this matches the demo.
        learning_rate = 1.0 - 0.9 * k / n_steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        history.append({"step": k, "loss": float(total_loss.data), "accuracy": float(acc)})

    curve_fig, axes = plt.subplots(2, 1, figsize=(5.2, 4.2), sharex=True)
    step_axis = [row["step"] for row in history]
    axes[0].plot(step_axis, [row["loss"] for row in history])
    axes[0].set_ylabel("loss")
    axes[1].plot(step_axis, [row["accuracy"] * 100 for row in history])
    axes[1].set_ylabel("accuracy %")
    axes[1].set_xlabel("step")
    axes[1].set_ylim(0, 105)
    curve_fig.tight_layout()

    # Decision boundary on a coarse grid, same resolution as the demo.
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in mesh]
    scores = list(map(model, inputs))
    grid = np.array([score.data > 0 for score in scores]).reshape(xx.shape)

    boundary_fig, boundary_ax = plt.subplots(figsize=(5, 5))
    boundary_ax.contourf(xx, yy, grid, cmap=plt.cm.Spectral, alpha=0.8)
    boundary_ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    boundary_ax.set_xlim(xx.min(), xx.max())
    boundary_ax.set_ylim(yy.min(), yy.max())
    boundary_ax.set_title("decision boundary")
    boundary_fig.tight_layout()

    first, last = history[0], history[-1]
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{last['loss']:.3f}",
                        label="损失",
                        caption=f"训练前 {first['loss']:.3f}",
                        bordered=True,
                        direction="decrease" if last["loss"] < first["loss"] else "increase",
                        target_direction="decrease",
                    ),
                    mo.stat(
                        f"{last['accuracy'] * 100:.0f}%",
                        label="准确率",
                        caption=f"训练前 {first['accuracy'] * 100:.0f}%",
                        bordered=True,
                        direction="increase"
                        if last["accuracy"] > first["accuracy"]
                        else "decrease",
                    ),
                ],
                justify="start",
                gap=1,
            ),
            curve_fig,
            boundary_fig,
        ]
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 训练后的同一张图

    权重换成这次 SGD 结束时的参数。梯度仍然只属于上面选中的那一条样本。
    """)
    return


@app.cell
def _(ModelGraph, X, mo, model, sample_index, trace_sample, view_height, y):
    trained_index = int(sample_index.value)
    trained_elements, trained_summary = trace_sample(
        model, X[trained_index], float(y[trained_index])
    )
    trained_graph = mo.ui.anywidget(
        ModelGraph(
            elements=trained_elements,
            summary=trained_summary,
            height=view_height(model),
        )
    )
    trained_graph
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 训练后的反向

    同一条样本、同一张布局，权重换成这次 SGD 结束时的参数。边上仍是这一条 hinge 的 $\partial w$。
    """)
    return


@app.cell
def _(ModelGraph, X, mo, model, sample_index, trace_sample, view_height, y):
    trained_back_index = int(sample_index.value)
    trained_back_elements, trained_back_summary = trace_sample(
        model, X[trained_back_index], float(y[trained_back_index]), view="backward"
    )
    trained_backward_graph = mo.ui.anywidget(
        ModelGraph(
            elements=trained_back_elements,
            summary=trained_back_summary,
            height=view_height(model),
            mode="backward",
        )
    )
    trained_backward_graph
    return


@app.cell
def _(X):
    X[0]
    return


if __name__ == "__main__":
    app.run()
