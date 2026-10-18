"""Initial controls for learning from an imperfect simulator."""

from __future__ import annotations

from configgle import PartialConfig

from priml.baselines.oscillator.benchmark import ForecastStudy
from priml.baselines.oscillator.calibration import CalibratedSimulator
from priml.baselines.oscillator.delayed import (
    DelayedCalibration,
    DelayedData,
    DelayedOscillator,
    DelayedStudy,
    SignedHistory,
)
from priml.baselines.oscillator.discovery import SparseSimulator
from priml.baselines.oscillator.forces import (
    CorrectedOscillator,
    PowerForce,
    WeightedForce,
)
from priml.baselines.oscillator.nonlinear import (
    HistoryAblation,
    KernelCorrection,
    NeuralForecast,
    PolynomialKernel,
)
from priml.baselines.oscillator.train_step import DLinearForecast, RidgeCorrection
from priml.model.init import unit_fan_in_uniform
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU


def exp000() -> ForecastStudy.Config:
    """DLinear trained only on the fixed reference sample.

    Hypothesis:
      An established forecaster gives a useful control before adding an agent.

    References:
      https://arxiv.org/abs/2205.13504
      priml.baselines.etth1.experiments.exp000 for the DLinear implementation.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = ForecastStudy.Config()
    cfg.experiment_name = "exp000"
    assert isinstance(cfg.method, DLinearForecast.Config)
    adam = cfg.method.step.optimizer
    assert isinstance(adam, PartialConfig)
    adam.lr = 1e-3
    return cfg


def exp001() -> ForecastStudy.Config:
    """exp000 with half its update budget spent on cheap simulator data.

    Hypothesis:
      Simulator pretraining may help when its dynamics match the reference,
      but may hurt when its missing force matters.

    References:
      exp000.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp000()
    cfg.experiment_name = "exp001"
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.pretrain_updates = cfg.method.updates // 2
    return cfg


def exp002() -> ForecastStudy.Config:
    """exp000 with ridge regression correcting the cheap simulator's forecasts.

    Hypothesis:
      A simple residual correction may solve this task without an agent.

    References:
      exp000 for the reference-data budget and held-out splits.
      https://doi.org/10.1080/00401706.1970.10488634

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp000()
    cfg.experiment_name = "exp002"
    cfg.method = RidgeCorrection.Config()
    return cfg


def exp003() -> ForecastStudy.Config:
    """exp002 with the cubic force fitted instead of an output correction.

    Hypothesis:
      Fitting a known force term on training transitions may hold up on larger
      displacements where a linear forecast correction fails. This gives the
      method the correct equation family, but not the reference coefficient.

    References:
      exp002 for the reference-data budget and residual-correction control.
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp002()
    cfg.experiment_name = "exp003"
    cfg.method = CalibratedSimulator.Config()
    return cfg


def exp004() -> ForecastStudy.Config:
    """exp003 with a reference force that cubic calibration cannot express.

    Hypothesis:
      Missing velocity-dependent drag will leave error after fitting a cubic
      restoring force, especially on larger displacements.

    References:
      exp003 for the calibration method and reference-data budget.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp003()
    cfg.experiment_name = "exp004"
    reference = cfg.data.reference = CorrectedOscillator.Config()
    reference.cubic = 0.4
    drag = WeightedForce.Config()
    drag.term = PowerForce.Config(axis=1, power=2)
    drag.coefficient = 0.35
    reference.corrections = [drag]
    return cfg


def exp005() -> ForecastStudy.Config:
    """exp004 with validation-selected force combinations instead of calibration.

    Hypothesis:
      Selecting missing restoring and drag terms may transfer better than
      adjusting a single known coefficient on the same reference sample.

    References:
      exp004.
      https://doi.org/10.1073/pnas.1517384113 for sparse equation discovery.
      This uses bounded nonlinear transition fits, not the paper's solver.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp004()
    cfg.experiment_name = "exp005"
    cfg.method = SparseSimulator.Config()
    return cfg


def exp006() -> ForecastStudy.Config:
    """exp005 without the correct drag term in its candidate library.

    Hypothesis:
      A library missing the true drag shape will expose the method's reliance
      on prior choices, even when validation selects a useful approximation.

    References:
      exp005.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp005()
    cfg.experiment_name = "exp006"
    assert isinstance(cfg.method, SparseSimulator.Config)
    cfg.method.terms = [
        PowerForce.Config(axis=0, power=1),
        PowerForce.Config(axis=0, power=3),
        PowerForce.Config(axis=0, power=5),
        PowerForce.Config(axis=1, power=1),
        PowerForce.Config(axis=1, power=3),
    ]
    return cfg


def exp007() -> ForecastStudy.Config:
    """exp004 with the original DLinear training recipe.

    Hypothesis:
      The same data-only forecaster gives a control for the harder dynamics.

    References:
      exp000 for the unchanged method; exp004 for the new reference system.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp004()
    cfg.experiment_name = "exp007"
    cfg.method = exp000().method
    return cfg


def exp008() -> ForecastStudy.Config:
    """exp004 with ridge correction of the simulator's outputs.

    Hypothesis:
      An output correction provides a simple control for learning force terms.

    References:
      exp002 for ridge; exp004 for the same reference system and data budget.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp004()
    cfg.experiment_name = "exp008"
    cfg.method = RidgeCorrection.Config()
    return cfg


def exp009() -> DelayedStudy.Config:
    """exp000 on a paired delayed-response task with nonlinear feedback.

    Hypothesis:
      Balanced pulse signs make the response orthogonal to every affine
      history predictor. A linear forecaster cannot beat the zero response.

    References:
      exp000 for the unchanged DLinear recipe and reference-data budget.
      https://proceedings.mlr.press/v235/toner24a.html for linear model classes.
      https://doi.org/10.1109/72.846741 for the NARMA delayed-memory benchmarks.
      This diagnostic uses prescribed histories, not the NARMA10 recurrence.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = DelayedStudy.Config()
    cfg.update(exp000())
    cfg.experiment_name = "exp009"
    cfg.data = DelayedData.Config()
    return cfg


def exp010() -> DelayedStudy.Config:
    """exp009 with a linear correction fitted to the full history.

    Hypothesis:
      Giving ridge every observed value still cannot express the missing
      interaction; a direct solve checks whether DLinear was undertrained.

    References:
      exp009; exp002 for the unchanged ridge method.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp009()
    cfg.experiment_name = "exp010"
    cfg.method = RidgeCorrection.Config()
    return cfg


def exp011() -> DelayedStudy.Config:
    """exp009 with the earlier cubic calibration control.

    Hypothesis:
      A model that discards the history cannot distinguish pulse pairs that
      end at the same physical state but produce opposite future responses.

    References:
      exp009; exp003 for the unchanged calibration method.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp009()
    cfg.experiment_name = "exp011"
    cfg.method = CalibratedSimulator.Config()
    return cfg


def exp012() -> DelayedStudy.Config:
    """exp009 with the earlier sparse force-selection control.

    Hypothesis:
      A richer memoryless force library will still lose the old pulse cues.

    References:
      exp009; exp005 for the unchanged force-selection method.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp009()
    cfg.experiment_name = "exp012"
    cfg.method = SparseSimulator.Config()
    return cfg


def exp013() -> DelayedStudy.Config:
    """exp010 with quadratic kernel ridge in place of an affine correction.

    Hypothesis:
      Pairwise history features can learn the sign interaction, but may
      extrapolate poorly when the true response saturates at larger inputs.

    References:
      exp010 for the same data and direct-solve control.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp010()
    cfg.experiment_name = "exp013"
    cfg.method = KernelCorrection.Config()
    return cfg


def exp014() -> DelayedStudy.Config:
    """exp009 with a gated nonlinear history network.

    Hypothesis:
      A network with multiplicative interactions can outperform DLinear on
      this response task at the same data and optimizer-update budgets.

    References:
      exp009; priml.model.swiglu.SwiGLU for the nonlinear block.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp009()
    cfg.experiment_name = "exp014"
    cfg.method = NeuralForecast.Config()
    return cfg


def exp015() -> DelayedStudy.Config:
    """exp009 with the correct delay equation and a training-fitted gain.

    Hypothesis:
      A privileged equation-aware control should forecast accurately, proving
      the visible history contains enough information for the task.

    References:
      exp009. This control knows both lags and the functional form.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp009()
    cfg.experiment_name = "exp015"
    assert isinstance(cfg.data.reference, DelayedOscillator.Config)
    method = cfg.method = DelayedCalibration.Config()
    method.simulator = cfg.data.reference.copy_tree()
    method.simulator.gain = 0.0
    return cfg


def exp016() -> DelayedStudy.Config:
    """exp014 with ten times as many optimizer updates.

    Hypothesis:
      More training separates an optimization limit from a representation limit.

    References:
      exp014.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp014()
    cfg.experiment_name = "exp016"
    assert isinstance(cfg.method, NeuralForecast.Config)
    cfg.method.updates *= 10
    return cfg


def exp017() -> DelayedStudy.Config:
    """exp016 with eight times the reference-data budget.

    Hypothesis:
      More independent magnitude pairs may help the nonlinear model learn
      saturation. This is a larger-budget control, not an equal-budget result.

    References:
      exp016. Training and validation both count toward the reference budget.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp016()
    cfg.experiment_name = "exp017"
    cfg.data.train.count *= 8
    cfg.data.validation.count *= 8
    return cfg


def exp018() -> DelayedStudy.Config:
    """exp015 with the informative older history masked during fit and prediction.

    Hypothesis:
      Even the correct equation cannot recover the response once both pulse
      cues are removed. This is an information ablation, not the main task.

    References:
      exp015.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp015()
    cfg.experiment_name = "exp018"
    cfg.method = HistoryAblation.Config(method=cfg.method)
    return cfg


def exp019() -> DelayedStudy.Config:
    """exp009 with four delayed factors and complete sixteen-sign groups.

    Hypothesis:
      A four-way response is orthogonal to all features of degree below four.
      Keep eight independent training magnitude tuples by expanding sign groups,
      rather than making the harder task use fewer independent examples.

    References:
      exp009 for the two-factor diagnostic and linear model control.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp009()
    cfg.experiment_name = "exp019"
    cfg.data.seq_len = 160
    assert isinstance(cfg.data, DelayedData.Config)
    cfg.data.history = SignedHistory.Config()
    assert isinstance(cfg.data.reference, DelayedOscillator.Config)
    cfg.data.reference.lags = [144, 112, 80, 48]
    cfg.data.train.count *= 4
    cfg.data.validation.count *= 4
    cfg.data.synthetic.count *= 4
    cfg.data.test.count *= 4
    cfg.data.shifted.count *= 4
    return cfg


def exp020() -> DelayedStudy.Config:
    """exp019 with the same affine ridge control.

    Hypothesis:
      A direct affine solve cannot recover the four-way response.

    References:
      exp019; exp010 for the same method on two factors.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp019()
    cfg.experiment_name = "exp020"
    cfg.method = RidgeCorrection.Config()
    return cfg


def exp021() -> DelayedStudy.Config:
    """exp019 with quadratic kernel correction.

    Hypothesis:
      Pairwise interactions that helped on two cues cannot express four-way
      parity, regardless of how well their coefficients are optimized.

    References:
      exp019; exp013 for the same nonlinear kernel on the two-factor task.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp019()
    cfg.experiment_name = "exp021"
    cfg.method = KernelCorrection.Config()
    return cfg


def exp022() -> DelayedStudy.Config:
    """exp019 with the same memoryless sparse force library.

    Hypothesis:
      Selecting more instantaneous forces cannot recover discarded history.

    References:
      exp019; exp012 for the same method on two factors.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp019()
    cfg.experiment_name = "exp022"
    cfg.method = SparseSimulator.Config()
    return cfg


def exp023() -> DelayedStudy.Config:
    """exp019 with the same nonlinear network and update budget.

    Hypothesis:
      A four-way dependency may be harder to learn than the earlier two-way
      dependency even when the number of independent training tuples is fixed.

    References:
      exp019; exp014 for the same nonlinear architecture and optimizer budget.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp019()
    cfg.experiment_name = "exp023"
    cfg.method = NeuralForecast.Config()
    return cfg


def exp024() -> DelayedStudy.Config:
    """exp019 with the known four-lag equation and a training-fitted gain.

    Hypothesis:
      The correct history-aware equation should still solve the task, despite
      low-degree or memoryless models failing on the same observations.

    References:
      exp019; exp015 for the privileged learnability control.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp019()
    cfg.experiment_name = "exp024"
    assert isinstance(cfg.data.reference, DelayedOscillator.Config)
    method = cfg.method = DelayedCalibration.Config()
    method.simulator = cfg.data.reference.copy_tree()
    method.simulator.gain = 0.0
    return cfg


def exp025() -> DelayedStudy.Config:
    """exp023 with ten times the optimizer updates.

    Hypothesis:
      A longer training run checks whether optimization explains a poor result.

    References:
      exp023.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp023()
    cfg.experiment_name = "exp025"
    assert isinstance(cfg.method, NeuralForecast.Config)
    cfg.method.updates *= 10
    return cfg


def exp026() -> DelayedStudy.Config:
    """exp021 with fourth-order instead of quadratic kernel features.

    Hypothesis:
      Supplying the missing interaction order should improve the response fit,
      isolating feature capacity from optimizer or information limitations.

    References:
      exp021. This control knows the interaction order, not the delayed lags.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp021()
    cfg.experiment_name = "exp026"
    assert isinstance(cfg.method, KernelCorrection.Config)
    assert isinstance(cfg.method.kernel, PolynomialKernel.Config)
    cfg.method.kernel.degree = 4
    return cfg


def exp027() -> DelayedStudy.Config:
    """exp025 with eight times as many reference histories.

    Hypothesis:
      Increasing independent magnitude tuples tests sample efficiency at the
      same update count. Its reference budget is larger than the main controls.

    References:
      exp025.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp025()
    cfg.experiment_name = "exp027"
    cfg.data.train.count *= 8
    cfg.data.validation.count *= 8
    return cfg


def exp028() -> DelayedStudy.Config:
    """exp025 with biases enabled in the single gated block.

    Hypothesis:
      Biases remove the quadratic-even-part restriction of one bias-free
      SwiGLU block. This checks architecture capacity at the same data budget.

    References:
      exp025; priml.model.swiglu.SwiGLU for the gate definition.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp025()
    cfg.experiment_name = "exp028"
    assert isinstance(cfg.method, NeuralForecast.Config)
    assert isinstance(cfg.method.step.model.block, SwiGLU.Config)
    cfg.method.step.model.block.bias = True
    return cfg


def exp029() -> DelayedStudy.Config:
    """exp025 with two composed gated blocks instead of one.

    Hypothesis:
      Composing nonlinear blocks can express the missing four-way interaction.
      The first block's output projection must start nonzero so both blocks
      receive gradients.

    References:
      exp025; exp028 for a separate capacity control using biases.

    Results:
      Reference-hardware results: TBD. Local pilot results are separate.

    """
    cfg = exp025()
    cfg.experiment_name = "exp029"
    assert isinstance(cfg.method, NeuralForecast.Config)
    cfg.method.step.model.block = Sequential.Config(
        elements=[
            SwiGLU.Config(
                channels_hidden=64,
                channels_out=64,
                init_weight_out=unit_fan_in_uniform,
            ),
            SwiGLU.Config(channels_hidden=64),
        ],
    )
    return cfg


def exp_smoke() -> ForecastStudy.Config:
    """Check the complete evaluation path with a small exp000 run.

    Hypothesis:
      Small shapes and two updates exercise the baseline wiring.

    References:
      exp000.

    Results:
      Integration only; not a forecasting result.

    """
    cfg = exp000()
    cfg.experiment_name = "exp_smoke"
    cfg.data.seq_len = 5
    cfg.data.pred_len = 3
    cfg.data.train.count = 4
    cfg.data.validation.count = 4
    cfg.data.synthetic.count = 6
    cfg.data.test.count = 4
    cfg.data.shifted.count = 4
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.updates = 2
    cfg.method.batch_size = 4
    return cfg
