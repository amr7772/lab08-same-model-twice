# PyTorch or Keras for the weekly bike demand job

For the manager and both team members. Short answer: **use PyTorch.**

## How I compared them

Both versions train the same network on the same prepared rows with the same loss, optimizer, learning rate, batch size and number of epochs, and both keep the best epoch's weights. With copied PyTorch weights, the Keras network's validation predictions matched to within 0.00000004 bikes, so both compute the same function. Three settings could not be made identical: the starting weights (Keras uses Glorot uniform weights and zero biases), the order of the shuffled batches, and Keras's default Adam epsilon, which I set to PyTorch's 1e-8. Each framework trained with seeds 42, 43 and 44 (`reports/comparison.csv`).

## What I observed

| | PyTorch | Keras |
|---|---|---|
| Test MAE, mean of 3 seeds | 2.41 | 2.47 |
| Seconds per run | 5.1 | 15.9 |
| Saved model file | 72 KB | 234 KB |

1. **Accuracy is the same in practice.** The test MAEs overlap across seeds (PyTorch 2.34 to 2.45, Keras 2.43 to 2.50), far below the linear regression's 8.76.
2. **Keras was about three times slower** on the same CPU. For a weekly job, neither time matters.
3. **Keras needs much less loop code.** My PyTorch `run_epoch` and `fit` are 41 lines I had to get right, plus 28 for the Dataset and loaders. In Keras the loop is one `fit` call, and two callbacks replace my best-weights code.
4. **Saving differs.** The `.keras` file holds the architecture, weights and optimizer state and loads on its own. The PyTorch file is smaller but needs `model.py` to rebuild it.
5. **Both in one environment clashed.** On Linux, loading TensorFlow first made the first PyTorch optimizer abort ("Option 'bitcode-mdindex-threshold' registered more than once") until I changed the import order.

## Recommendation

PyTorch. The Hugging Face plan decides it: Transformers v5 dropped TensorFlow support and now supports PyTorch only, so a Keras pipeline would need rewriting when the team tries those models. The PyTorch version is also already tested, and one framework avoids the clash in point 5.

## What the team gives up

The shorter Keras code, the ready-made callbacks, the single-file model save, and the Keras member's head start. Someone must keep the hand-written loop and its tests correct.

## What would change my recommendation

If the team drops the Hugging Face plan and the Keras member owns the job, I would choose Keras: less code to maintain, and ten extra seconds a week costs nothing.
