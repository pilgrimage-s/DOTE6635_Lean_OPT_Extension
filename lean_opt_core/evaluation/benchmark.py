"""
Benchmark experiment runner for comparing original vs orchestrator-enhanced pipelines.

Measures:
- Modeling Accuracy (EM)
- Optimal Value Accuracy
- Execution Time
- Token Usage
- Success Rate
"""

from typing import List, Dict, Any, Optional, Callable
import json
import time
from dataclasses import asdict, dataclass
import traceback

from ..core.types import ExperimentResult, BenchmarkResult, PipelineOutput
from ..evaluation.metrics import ModelingAccuracy, FormulationMatcher, OptimalValueValidator
from ..core.pipeline import LeanOptPipeline, PipelineComparator


@dataclass
class BenchmarkProblem:
    """Represents a single benchmark problem instance."""
    problem_id: str
    problem_description: str
    problem_type: str
    datasets: List[str]
    ground_truth_formulation: str
    ground_truth_optimal_value: Optional[float]
    example_code: Optional[str] = None


class BenchmarkRunner:
    """
    Runs experiments comparing original vs orchestrator-enhanced pipelines.
    
    Workflow:
    1. Load benchmark dataset (problems with ground truth formulations)
    2. For each problem:
       - Run original pipeline → get formulation
       - Run enhanced pipeline → get formulation + verifications
       - Compare to ground truth
       - Record metrics
    3. Aggregate results and generate report
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4",
        verbose: bool = False,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            api_key: OpenAI API key
            model_name: LLM model name
            verbose: Print detailed output
        """
        self.api_key = api_key
        self.model_name = model_name
        self.verbose = verbose

        self.original_pipeline = LeanOptPipeline(
            api_key=api_key,
            model_name=model_name,
            use_orchestrator=False,
            verbose=verbose,
        )

        self.enhanced_pipeline = LeanOptPipeline(
            api_key=api_key,
            model_name=model_name,
            use_orchestrator=True,
            verbose=verbose,
        )

    def run_single_experiment(
        self,
        problem: BenchmarkProblem,
        pipeline_output_getter: Callable,
    ) -> ExperimentResult:
        """
        Run single experiment on one problem.
        
        Args:
            problem: Benchmark problem instance
            pipeline_output_getter: Function that generates pipeline output
                                   Returns: (classification, workflow, formulation, code)
                                   
        Returns:
            ExperimentResult with metrics
        """
        start_time = time.time()
        errors = []

        try:
            # Get pipeline outputs
            classification, workflow, formulation, code = pipeline_output_getter(
                problem.problem_description
            )

            # Check if formulation matches ground truth
            formulation_matches = FormulationMatcher.exact_match(
                formulation, problem.ground_truth_formulation, tolerance=0.15
            )

            # Check if optimal value matches (if available)
            optimal_matches = False
            if problem.ground_truth_optimal_value is not None and code:
                optimal_value = OptimalValueValidator.extract_optimal_value(code)
                optimal_matches = OptimalValueValidator.compare_optimal_values(
                    optimal_value, problem.ground_truth_optimal_value
                )

            execution_time = time.time() - start_time

            result = ExperimentResult(
                problem_id=problem.problem_id,
                problem_type=problem.problem_type,
                use_orchestrator=False,  # Set by caller
                original_output=None,  # Set by caller
                formulation_matches_ground_truth=formulation_matches,
                optimal_value_matches=optimal_matches,
                execution_time=execution_time,
                errors=errors,
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            errors.append(f"{type(e).__name__}: {str(e)}")
            if self.verbose:
                traceback.print_exc()

            return ExperimentResult(
                problem_id=problem.problem_id,
                problem_type=problem.problem_type,
                use_orchestrator=False,
                original_output=None,
                formulation_matches_ground_truth=False,
                optimal_value_matches=False,
                execution_time=execution_time,
                errors=errors,
            )

    def run_benchmark(
        self,
        problems: List[BenchmarkProblem],
        original_output_generator: Callable,
        enhanced_output_generator: Callable,
    ) -> tuple:
        """
        Run full benchmark comparing both pipelines.
        
        Args:
            problems: List of benchmark problems
            original_output_generator: Function(problem_description) -> (classification, workflow, formulation, code)
            enhanced_output_generator: Function(problem_description) -> (classification, workflow, formulation, code)
            
        Returns:
            Tuple of (original_results, enhanced_results)
        """
        original_results = []
        enhanced_results = []

        total_problems = len(problems)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"BENCHMARK RUNNER: Testing {total_problems} problems")
            print(f"{'='*70}\n")

        for idx, problem in enumerate(problems):
            if self.verbose:
                print(
                    f"[{idx+1}/{total_problems}] Problem: {problem.problem_id} "
                    f"({problem.problem_type})"
                )

            # Run original pipeline
            if self.verbose:
                print("  Running original pipeline...")
            
            original_result = self.run_single_experiment(
                problem, original_output_generator
            )
            original_result.use_orchestrator = False
            original_results.append(original_result)

            if self.verbose:
                print(f"    EM Score: {original_result.em_score} "
                      f"(Formulation match: {original_result.formulation_matches_ground_truth})")

            # Run enhanced pipeline
            if self.verbose:
                print("  Running enhanced pipeline...")

            enhanced_result = self.run_single_experiment(
                problem, enhanced_output_generator
            )
            enhanced_result.use_orchestrator = True
            enhanced_results.append(enhanced_result)

            if self.verbose:
                print(f"    EM Score: {enhanced_result.em_score} "
                      f"(Formulation match: {enhanced_result.formulation_matches_ground_truth})")
                print()

        return original_results, enhanced_results

    def generate_report(
        self,
        original_results: List[ExperimentResult],
        enhanced_results: List[ExperimentResult],
    ) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            original_results: Results from original pipeline
            enhanced_results: Results from enhanced pipeline
            
        Returns:
            Formatted report string
        """
        original_benchmark = BenchmarkResult(
            total_problems=len(original_results),
            use_orchestrator=False,
            results=original_results,
        )

        enhanced_benchmark = BenchmarkResult(
            total_problems=len(enhanced_results),
            use_orchestrator=True,
            results=enhanced_results,
        )

        # Calculate improvements
        em_improvement = enhanced_benchmark.em_accuracy - original_benchmark.em_accuracy
        optimal_improvement = (
            enhanced_benchmark.optimal_accuracy - original_benchmark.optimal_accuracy
        )
        time_difference = enhanced_benchmark.average_time - original_benchmark.average_time

        report = f"""

{'='*70}
LEAN-LLM-OPT BENCHMARK RESULTS
{'='*70}

{original_benchmark.summary()}

{enhanced_benchmark.summary()}

{'='*70}
COMPARATIVE ANALYSIS
{'='*70}

Modeling Accuracy (EM):
  Original:                 {original_benchmark.em_accuracy:>6.2f}%
  Enhanced (Orchestrator):  {enhanced_benchmark.em_accuracy:>6.2f}%
  Improvement:              {em_improvement:>+6.2f}% {"✓" if em_improvement > 0 else "✗"}

Optimal Value Accuracy:
  Original:                 {original_benchmark.optimal_accuracy:>6.2f}%
  Enhanced (Orchestrator):  {enhanced_benchmark.optimal_accuracy:>6.2f}%
  Improvement:              {optimal_improvement:>+6.2f}% {"✓" if optimal_improvement > 0 else "✗"}

Average Execution Time:
  Original:                 {original_benchmark.average_time:>6.2f}s
  Enhanced (Orchestrator):  {enhanced_benchmark.average_time:>6.2f}s
  Difference:               {time_difference:>+6.2f}s {"(slower)" if time_difference > 0 else "(faster)"}

Success Rate:
  Original:                 {original_benchmark.success_rate:>6.2f}%
  Enhanced (Orchestrator):  {enhanced_benchmark.success_rate:>6.2f}%
  Difference:               {enhanced_benchmark.success_rate - original_benchmark.success_rate:>+6.2f}%

{'='*70}
DETAILED RESULTS BY PROBLEM TYPE
{'='*70}

"""
        # Group by problem type
        type_stats = {}
        
        for orig, enh in zip(original_results, enhanced_results):
            ptype = orig.problem_type
            if ptype not in type_stats:
                type_stats[ptype] = {
                    "original_em": [],
                    "enhanced_em": [],
                    "count": 0,
                }
            type_stats[ptype]["original_em"].append(orig.em_score)
            type_stats[ptype]["enhanced_em"].append(enh.em_score)
            type_stats[ptype]["count"] += 1

        for ptype in sorted(type_stats.keys()):
            stats = type_stats[ptype]
            orig_avg = (
                sum(stats["original_em"]) / len(stats["original_em"])
                if stats["original_em"]
                else 0
            )
            enh_avg = (
                sum(stats["enhanced_em"]) / len(stats["enhanced_em"])
                if stats["enhanced_em"]
                else 0
            )
            improvement = enh_avg - orig_avg

            report += f"""
{ptype:>4} ({stats['count']:>2} problems):
  Original EM:      {orig_avg*100:>6.2f}%
  Enhanced EM:      {enh_avg*100:>6.2f}%
  Improvement:      {improvement*100:>+6.2f}%
"""

        report += f"\n{'='*70}\n"

        return report

    def save_results(
        self,
        original_results: List[ExperimentResult],
        enhanced_results: List[ExperimentResult],
        filepath: str,
    ) -> None:
        """Save results to JSON file."""
        data = {
            "original": {
                "use_orchestrator": False,
                "total": len(original_results),
                "results": [r.to_dict() for r in original_results],
            },
            "enhanced": {
                "use_orchestrator": True,
                "total": len(enhanced_results),
                "results": [r.to_dict() for r in enhanced_results],
            },
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"[BenchmarkRunner] Results saved to {filepath}")
