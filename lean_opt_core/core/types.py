"""Data types and structures for LEAN-LLM-OPT pipeline."""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json


class ProblemType(str, Enum):
    """Problem classification types."""
    NRM = "NRM"  # Network Revenue Management
    RA = "RA"    # Resource Allocation
    TP = "TP"    # Transportation Problem
    FLP = "FLP"  # Facility Location Problem
    AP = "AP"    # Assignment Problem
    SBLP = "SBLP"  # Sales-Based Linear Programming
    MIXTURE = "Mixture"  # Combination of multiple types
    OTHERS = "Others"  # Novel or uncategorized


class VerificationStatus(str, Enum):
    """Verification status from orchestrator."""
    PASS = "PASS"
    NEEDS_REFINEMENT = "NEEDS_REFINEMENT"
    CRITICAL_FLAW = "CRITICAL_FLAW"


class FlawSeverity(str, Enum):
    """Severity levels for identified flaws."""
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


@dataclass
class Flaw:
    """Represents a single flaw identified by orchestrator."""
    severity: FlawSeverity
    issue: str
    location: str  # Component where flaw exists
    root_cause: str
    fix: str


@dataclass
class WorkflowVerification:
    """Result of workflow verification by orchestrator."""
    status: VerificationStatus
    follows_template: bool
    data_retrieval_appropriate: bool
    components_complete: bool
    will_guide_model_generation: bool
    flaws: List[Flaw]
    confidence: str  # "High", "Medium", "Low"
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "follows_template": self.follows_template,
            "data_retrieval_appropriate": self.data_retrieval_appropriate,
            "components_complete": self.components_complete,
            "will_guide_model_generation": self.will_guide_model_generation,
            "flaws": [
                {
                    "severity": f.severity.value,
                    "issue": f.issue,
                    "location": f.location,
                    "root_cause": f.root_cause,
                    "fix": f.fix,
                }
                for f in self.flaws
            ],
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class ModelVerification:
    """Result of model verification by orchestrator."""
    status: VerificationStatus
    consistency_with_workflow: bool
    completeness: bool
    mathematical_soundness: bool
    problem_alignment: bool
    data_integrity: bool
    flaws: List[Flaw]
    confidence: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "consistency_with_workflow": self.consistency_with_workflow,
            "completeness": self.completeness,
            "mathematical_soundness": self.mathematical_soundness,
            "problem_alignment": self.problem_alignment,
            "data_integrity": self.data_integrity,
            "flaws": [
                {
                    "severity": f.severity.value,
                    "issue": f.issue,
                    "location": f.location,
                    "root_cause": f.root_cause,
                    "fix": f.fix,
                }
                for f in self.flaws
            ],
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class CodeVerification:
    """Result of code verification by orchestrator."""
    status: VerificationStatus
    syntax_valid: bool
    api_calls_correct: bool
    model_initialization_proper: bool
    constraint_formulation_matches_math: bool
    data_types_consistent: bool
    flaws: List[Flaw]
    confidence: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "syntax_valid": self.syntax_valid,
            "api_calls_correct": self.api_calls_correct,
            "model_initialization_proper": self.model_initialization_proper,
            "constraint_formulation_matches_math": self.constraint_formulation_matches_math,
            "data_types_consistent": self.data_types_consistent,
            "flaws": [
                {
                    "severity": f.severity.value,
                    "issue": f.issue,
                    "location": f.location,
                    "root_cause": f.root_cause,
                    "fix": f.fix,
                }
                for f in self.flaws
            ],
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass
class PipelineOutput:
    """Final output from LEAN-LLM-OPT pipeline."""
    problem_type: str
    user_query: str
    classification_result: str
    workflow: str
    workflow_verification: Optional[WorkflowVerification] = None
    formulation: str = ""
    model_verification: Optional[ModelVerification] = None
    code: str = ""
    code_verification: Optional[CodeVerification] = None
    status: str = "success"
    num_refinement_iterations: int = 0
    use_orchestrator: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem_type": self.problem_type,
            "user_query": self.user_query,
            "classification_result": self.classification_result,
            "workflow": self.workflow,
            "workflow_verification": (
                self.workflow_verification.to_dict()
                if self.workflow_verification
                else None
            ),
            "formulation": self.formulation,
            "model_verification": (
                self.model_verification.to_dict()
                if self.model_verification
                else None
            ),
            "code": self.code,
            "code_verification": (
                self.code_verification.to_dict()
                if self.code_verification
                else None
            ),
            "status": self.status,
            "num_refinement_iterations": self.num_refinement_iterations,
            "use_orchestrator": self.use_orchestrator,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""
    problem_id: str
    problem_type: str
    use_orchestrator: bool
    original_output: PipelineOutput
    formulation_matches_ground_truth: bool
    optimal_value_matches: bool
    execution_time: float  # seconds
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def em_score(self) -> int:
        """Modeling accuracy (EM): 1 if formulation matches, 0 otherwise."""
        return 1 if self.formulation_matches_ground_truth else 0

    @property
    def optimal_accuracy(self) -> int:
        """Optimal value accuracy: 1 if matches, 0 otherwise."""
        return 1 if self.optimal_value_matches else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem_id": self.problem_id,
            "problem_type": self.problem_type,
            "use_orchestrator": self.use_orchestrator,
            "formulation_matches_ground_truth": self.formulation_matches_ground_truth,
            "optimal_value_matches": self.optimal_value_matches,
            "em_score": self.em_score,
            "optimal_accuracy": self.optimal_accuracy,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "errors": self.errors,
        }


@dataclass
class BenchmarkResult:
    """Aggregated results from benchmark experiment."""
    total_problems: int
    use_orchestrator: bool
    results: List[ExperimentResult]
    
    @property
    def em_accuracy(self) -> float:
        """Modeling accuracy (EM) percentage."""
        if not self.results:
            return 0.0
        return (sum(r.em_score for r in self.results) / len(self.results)) * 100

    @property
    def optimal_accuracy(self) -> float:
        """Optimal value accuracy percentage."""
        if not self.results:
            return 0.0
        return (
            sum(r.optimal_accuracy for r in self.results) / len(self.results)
        ) * 100

    @property
    def average_time(self) -> float:
        """Average execution time per problem."""
        if not self.results:
            return 0.0
        return sum(r.execution_time for r in self.results) / len(self.results)

    @property
    def total_time(self) -> float:
        """Total execution time."""
        return sum(r.execution_time for r in self.results)

    @property
    def success_rate(self) -> float:
        """Percentage of problems executed without errors."""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if not r.errors)
        return (successful / len(self.results)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_problems": self.total_problems,
            "use_orchestrator": self.use_orchestrator,
            "em_accuracy": self.em_accuracy,
            "optimal_accuracy": self.optimal_accuracy,
            "average_time": self.average_time,
            "total_time": self.total_time,
            "success_rate": self.success_rate,
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        mode = "WITH ORCHESTRATOR" if self.use_orchestrator else "ORIGINAL PIPELINE"
        return f"""
╔════════════════════════════════════════════════════════════╗
║          BENCHMARK RESULTS - {mode:^35} ║
╠════════════════════════════════════════════════════════════╣
║  Total Problems:          {self.total_problems:>4d}                        ║
║  Modeling Accuracy (EM):  {self.em_accuracy:>6.2f}%                        ║
║  Optimal Value Accuracy:  {self.optimal_accuracy:>6.2f}%                        ║
║  Success Rate:            {self.success_rate:>6.2f}%                        ║
║  Average Time/Problem:    {self.average_time:>6.2f}s                        ║
║  Total Time:              {self.total_time:>6.2f}s                        ║
╚════════════════════════════════════════════════════════════╝
"""
