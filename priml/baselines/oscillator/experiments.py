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
from priml.baselines.oscillator.mechanisms import (
    CoupledDrag,
    SaturatingForce,
    SineForce,
    SystemFamily,
)
from priml.baselines.oscillator.nonlinear import (
    HistoryAblation,
    KernelCorrection,
    NeuralForecast,
    PolynomialKernel,
)
from priml.baselines.oscillator.pretrained import Chronos2Pipeline, PretrainedForecast
from priml.baselines.oscillator.sequence import (
    PatchSequence,
    RbfKernel,
    RecurrentSequence,
)
from priml.baselines.oscillator.suite import (
    BenchmarkRecipe,
    BenchmarkSuite,
    BudgetStudy,
    ReferenceBudget,
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
      https://maths-people.anu.edu.au/~brent/pub/pub011.html for the search.

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


def exp030() -> BudgetStudy.Config:
    """exp000 with the transfer benchmark's longer context and data protocol.

    Hypothesis:
      A full observed cycle and independent trajectories give each model a
      meaningful chance to infer the dynamics. Geometry and split counts define
      this new task together; the earlier baseline remains unchanged.

    References:
      exp000; PROTOCOL.md for the version 0.1 task definition.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = BudgetStudy.Config()
    cfg.update(exp000())
    cfg.experiment_name = "exp030"
    cfg.data.seq_len = 64
    cfg.data.pred_len = 24
    cfg.data.cheap.substeps = 8
    cfg.data.reference.substeps = cfg.data.cheap.substeps
    cfg.data.train.count = 64
    cfg.data.validation.count = 32
    cfg.data.test.count = 128
    cfg.data.shifted.count = cfg.data.test.count
    return cfg


def exp031() -> BudgetStudy.Config:
    """exp030 with eight times as many optimizer updates.

    Hypothesis:
      A 2,400-update control reduces the chance of mistaking short training for
      a model limitation on the longer context.

    References:
      exp030.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp030()
    cfg.experiment_name = "exp031"
    assert isinstance(cfg.method, DLinearForecast.Config)
    cfg.method.updates *= 8
    return cfg


def exp032() -> BudgetStudy.Config:
    """exp031 with the existing affine simulator correction.

    Hypothesis:
      A direct solve tests whether learning a linear output correction is enough.

    References:
      exp031; exp002 for the fixed ridge penalty.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp031()
    cfg.experiment_name = "exp032"
    cfg.method = RidgeCorrection.Config()
    return cfg


def exp033() -> BudgetStudy.Config:
    """exp032 with a smooth nonlinear kernel instead of an affine correction.

    Hypothesis:
      A radial kernel gives a flexible direct-solve control, with no selected
      polynomial interaction order or audit-dependent tuning.

    References:
      exp032; KernelCorrection for training-only feature normalization.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp032()
    cfg.experiment_name = "exp033"
    method = cfg.method = KernelCorrection.Config()
    method.kernel = RbfKernel.Config()
    return cfg


def exp034() -> BudgetStudy.Config:
    """exp031 with the fixed sparse force-library search.

    Hypothesis:
      A known force library may transfer on familiar terms and their unseen
      composition, while missing entirely new force shapes.

    References:
      exp031; exp005 for the library and validation rule.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp031()
    cfg.experiment_name = "exp034"
    cfg.method = SparseSimulator.Config()
    return cfg


def exp035() -> BudgetStudy.Config:
    """exp031 with two nonlinear history blocks at the same update budget.

    Hypothesis:
      A model capable of higher-order interactions is a stronger neural control
      than the single bias-free block used in the earlier diagnostic.

    References:
      exp031; exp029 for the gradient-preserving two-block initialization.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp031()
    cfg.experiment_name = "exp035"
    parent = cfg.method
    assert isinstance(parent, DLinearForecast.Config)
    method = cfg.method = NeuralForecast.Config()
    method.updates = parent.updates
    block = method.step.model.block = Sequential.Config()
    first = SwiGLU.Config()
    first.channels_hidden = 64
    first.channels_out = 64
    first.bias = True
    first.init_weight_out = unit_fan_in_uniform
    second = SwiGLU.Config()
    second.channels_hidden = 64
    second.bias = True
    block.elements = [first, second]
    return cfg


def exp036() -> BudgetStudy.Config:
    """exp035 with patch attention in place of the flat history blocks.

    Hypothesis:
      Explicit temporal mixing may improve sample efficiency on unseen dynamics.

    References:
      exp035; priml.model.transformer.block.TransformerBlock.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp035()
    cfg.experiment_name = "exp036"
    assert isinstance(cfg.method, NeuralForecast.Config)
    cfg.method.step.model.block = PatchSequence.Config()
    return cfg


def exp037() -> BudgetStudy.Config:
    """exp036 with two recurrent layers instead of the attention stack.

    Hypothesis:
      Recurrence supplies another sequence-model control at the same data and
      optimizer-update budgets.

    References:
      exp036; https://arxiv.org/abs/2410.01201 for minGRU.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp036()
    cfg.experiment_name = "exp037"
    assert isinstance(cfg.method, NeuralForecast.Config)
    assert isinstance(cfg.method.step.model.block, PatchSequence.Config)
    cfg.method.step.model.block.block = RecurrentSequence.Config()
    return cfg


def exp038() -> BudgetStudy.Config:
    """exp031 with pinned zero-shot Chronos-Bolt forecasts.

    Hypothesis:
      A pretrained forecaster provides an external anchor without using any
      reference adaptation targets. Its two channels are forecast independently.

    References:
      exp031; https://github.com/amazon-science/chronos-forecasting.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp031()
    cfg.experiment_name = "exp038"
    cfg.method = PretrainedForecast.Config()
    return cfg


def exp039() -> BudgetStudy.Config:
    """exp038 with a pinned multivariate Chronos-2 pipeline.

    Hypothesis:
      A stronger pretrained model that keeps both channels in one group may
      improve over the small univariate pretrained control.

    References:
      exp038; https://arxiv.org/abs/2510.15821.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = exp038()
    cfg.experiment_name = "exp039"
    assert isinstance(cfg.method, PretrainedForecast.Config)
    cfg.method.pipeline = Chronos2Pipeline.Config()
    return cfg


def exp040() -> BenchmarkSuite.Config:
    """Freeze exp031--exp039 on the version 0.1 mechanism-holdout suite.

    Hypothesis:
      Comparing adaptation across held-out force laws and two reference budgets
      separates interpolation from correction that transfers to new conditions.

    References:
      PROTOCOL.md; exp031--exp039 for every individual method recipe.

    Results:
      Reference-hardware results: TBD. Local results are reported separately.

    """
    cfg = BenchmarkSuite.Config()
    cfg.data = exp031().data
    large = ReferenceBudget.Config()
    large.name = "large"
    large.train *= 4
    large.validation *= 4
    cfg.budgets = [ReferenceBudget.Config(), large]

    matched = SystemFamily.Config()
    matched.name = "matched"
    matched.instances = 1
    cubic = SystemFamily.Config()
    cubic.name = "cubic"
    cubic.seed = 12_001
    cubic.instances = 1
    cubic.reference.cubic = 0.4
    drag = SystemFamily.Config()
    drag.name = "quadratic_drag"
    drag.seed = 13_001
    drag.instances = 1
    drag_force = WeightedForce.Config()
    drag_term = drag_force.term = PowerForce.Config()
    drag_term.axis = 1
    drag_term.power = 2
    drag_force.coefficient = 0.35
    drag.reference.corrections = [drag_force]
    cfg.development = [matched, cubic, drag]

    mixed = SystemFamily.Config()
    mixed.name = "combined"
    mixed.seed = 21_001
    mixed.reference = cubic.reference.copy_tree()
    mixed.reference.corrections = [
        correction.copy_tree() for correction in drag.reference.corrections
    ]
    bounded = SystemFamily.Config()
    bounded.name = "saturating"
    bounded.seed = 22_001
    bounded_force = WeightedForce.Config()
    bounded_force.term = SaturatingForce.Config()
    bounded_force.coefficient = 1.2
    bounded.reference.corrections = [bounded_force]
    sine = SystemFamily.Config()
    sine.name = "sine"
    sine.seed = 23_001
    sine_force = WeightedForce.Config()
    sine_force.term = SineForce.Config()
    sine_force.coefficient = 0.8
    sine.reference.corrections = [sine_force]
    coupled = SystemFamily.Config()
    coupled.name = "coupled_drag"
    coupled.seed = 24_001
    coupled_force = WeightedForce.Config()
    coupled_force.term = CoupledDrag.Config()
    coupled_force.coefficient = 0.45
    coupled.reference.corrections = [coupled_force]
    cfg.audit = [mixed, bounded, sine, coupled]

    dlinear = BenchmarkRecipe.Config()
    dlinear.study = exp031()
    ridge = BenchmarkRecipe.Config()
    ridge.name = "ridge"
    ridge.study = exp032()
    ridge.seeds = [2021]
    kernel = BenchmarkRecipe.Config()
    kernel.name = "rbf"
    kernel.study = exp033()
    kernel.seeds = [2021]
    forces = BenchmarkRecipe.Config()
    forces.name = "force_library"
    forces.study = exp034()
    forces.seeds = [2021]
    dense = BenchmarkRecipe.Config()
    dense.name = "history_net"
    dense.study = exp035()
    attention = BenchmarkRecipe.Config()
    attention.name = "patch_attention"
    attention.study = exp036()
    recurrent = BenchmarkRecipe.Config()
    recurrent.name = "patch_mingru"
    recurrent.study = exp037()
    bolt = BenchmarkRecipe.Config()
    bolt.name = "chronos_bolt_tiny"
    bolt.study = exp038()
    bolt.seeds = [2021]
    bolt.access = "zero-shot"
    chronos2 = BenchmarkRecipe.Config()
    chronos2.name = "chronos2"
    chronos2.study = exp039()
    chronos2.seeds = [2021]
    chronos2.access = "zero-shot"
    cfg.recipes = [
        dlinear,
        ridge,
        kernel,
        forces,
        dense,
        attention,
        recurrent,
        bolt,
        chronos2,
    ]
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
