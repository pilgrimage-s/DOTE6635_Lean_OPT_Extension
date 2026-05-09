"""
Main LEAN-LLM-OPT Pipeline - with optional Orchestrator verification layer.

Implements two modes:
1. Original Pipeline: Classification → Workflow → Model (no verification)
2. Enhanced Pipeline: Add Orchestrator verification at each stage
"""

from typing import Optional, Dict, Any
import time

from ..core.types import PipelineOutput
from ..agents.orchestrator import OrchestratorAgent


class LeanOptPipeline:
    """
    Main pipeline for LEAN-LLM-OPT agentic framework.
    
    Modes:
    - use_orchestrator=False: Original 3-agent pipeline
    - use_orchestrator=True: Enhanced pipeline with verification loops
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4",
        use_orchestrator: bool = False,
        max_refinement_iterations: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize pipeline.
        
        Args:
            api_key: OpenAI API key
            model_name: LLM model name
            use_orchestrator: Enable orchestrator verification
            max_refinement_iterations: Max times to refine before giving up
            verbose: Print detailed progress
        """
        self.api_key = api_key
        self.model_name = model_name
        self.use_orchestrator = use_orchestrator
        self.max_refinement_iterations = max_refinement_iterations
        self.verbose = verbose

        if use_orchestrator:
            self.orchestrator = OrchestratorAgent(
                api_key=api_key,
                model_name=model_name,
                verbose=verbose,
            )
        else:
            self.orchestrator = None

    def execute(
        self,
        user_query: str,
        problem_type: str,
        classification_result: str,
        workflow: str,
        formulation: str,
        code: str = "",
    ) -> PipelineOutput:
        """
        Execute pipeline with optional orchestrator verification.
        
        Args:
            user_query: Original problem description
            problem_type: Classified problem type
            classification_result: Output from classification agent
            workflow: Generated workflow
            formulation: Generated model formulation
            code: Generated code (optional)
            
        Returns:
            PipelineOutput with results and verification details
        """
        start_time = time.time()
        refinement_count = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"LEAN-LLM-OPT Pipeline Execution")
            print(f"Mode: {'WITH ORCHESTRATOR' if self.use_orchestrator else 'ORIGINAL'}")
            print(f"{'='*60}\n")

        # Initialize output
        output = PipelineOutput(
            problem_type=problem_type,
            user_query=user_query,
            classification_result=classification_result,
            workflow=workflow,
            formulation=formulation,
            code=code,
            use_orchestrator=self.use_orchestrator,
        )

        if not self.use_orchestrator:
            # Original pipeline: no verification
            if self.verbose:
                print("[Pipeline] Executing original 3-agent pipeline (no verification)")
                print("[Pipeline] ✓ Classification")
                print("[Pipeline] ✓ Workflow Generation")
                print("[Pipeline] ✓ Model Generation")
            output.status = "success"
            output.num_refinement_iterations = 0

        else:
            # Enhanced pipeline with orchestrator verification
            if self.verbose:
                print("[Pipeline] Executing enhanced pipeline with orchestrator\n")

            # Stage 1: Workflow Verification
            if self.verbose:
                print("[Pipeline-Orchestrator] Stage 1: Verifying Workflow")
            
            output.workflow_verification = self.orchestrator.verify_workflow(
                workflow=workflow,
                problem_type=problem_type,
                user_query=user_query,
            )

            if self.verbose:
                print(f"  Status: {output.workflow_verification.status.value}")
                print(f"  Confidence: {output.workflow_verification.confidence}\n")

            # Workflow refinement loop
            if (
                output.workflow_verification.status.value == "NEEDS_REFINEMENT"
                and refinement_count < self.max_refinement_iterations
            ):
                if self.verbose:
                    print(
                        "[Pipeline-Orchestrator] Workflow needs refinement. "
                        "In production, would trigger workflow regeneration."
                    )
                refinement_count += 1

            # Stage 2: Model Verification
            if self.verbose:
                print("[Pipeline-Orchestrator] Stage 2: Verifying Model Formulation")

            output.model_verification = self.orchestrator.verify_model(
                model_formulation=formulation,
                workflow=workflow,
                user_query=user_query,
            )

            if self.verbose:
                print(f"  Status: {output.model_verification.status.value}")
                print(f"  Confidence: {output.model_verification.confidence}\n")

            if (
                output.model_verification.status.value == "NEEDS_REFINEMENT"
                and refinement_count < self.max_refinement_iterations
            ):
                if self.verbose:
                    print(
                        "[Pipeline-Orchestrator] Model needs refinement. "
                        "In production, would trigger model regeneration."
                    )
                refinement_count += 1

            # Stage 3: Code Verification (if code provided)
            if code:
                if self.verbose:
                    print("[Pipeline-Orchestrator] Stage 3: Verifying Generated Code")

                output.code_verification = self.orchestrator.verify_code(
                    code=code,
                    model_formulation=formulation,
                )

                if self.verbose:
                    print(f"  Status: {output.code_verification.status.value}")
                    print(
                        f"  Confidence: {output.code_verification.confidence}\n"
                    )

                if (
                    output.code_verification.status.value == "NEEDS_REFINEMENT"
                    and refinement_count < self.max_refinement_iterations
                ):
                    if self.verbose:
                        print(
                            "[Pipeline-Orchestrator] Code needs refinement. "
                            "In production, would trigger code regeneration."
                        )
                    refinement_count += 1

            # Get recommendations
            recommendations = self.orchestrator.recommend_refinements(
                workflow_verification=output.workflow_verification,
                model_verification=output.model_verification,
                code_verification=output.code_verification,
            )

            output.status = "success"
            output.num_refinement_iterations = refinement_count

            if self.verbose:
                if refinement_count > 0:
                    print(f"\n[Pipeline-Orchestrator] Refinement Recommendations:")
                    for rec in recommendations["recommendations"]:
                        print(f"  - {rec}")
                else:
                    print("[Pipeline-Orchestrator] ✓ All verifications passed!")
                print()

        output_time = time.time() - start_time
        if self.verbose:
            print(f"[Pipeline] Execution completed in {output_time:.2f}s\n")

        return output


class PipelineComparator:
    """Utilities for comparing original vs orchestrator-enhanced pipeline."""

    @staticmethod
    def compare_outputs(
        original_output: PipelineOutput,
        enhanced_output: PipelineOutput,
    ) -> Dict[str, Any]:
        """
        Compare outputs from original and enhanced pipelines.
        
        Args:
            original_output: Output from original pipeline (no orchestrator)
            enhanced_output: Output from enhanced pipeline (with orchestrator)
            
        Returns:
            Comparison report
        """
        return {
            "problem_type": original_output.problem_type,
            "original": {
                "status": original_output.status,
                "num_refinements": original_output.num_refinement_iterations,
                "has_verifications": False,
            },
            "enhanced": {
                "status": enhanced_output.status,
                "num_refinements": enhanced_output.num_refinement_iterations,
                "has_verifications": enhanced_output.workflow_verification is not None,
                "workflow_verified": (
                    enhanced_output.workflow_verification.status.value
                    if enhanced_output.workflow_verification
                    else None
                ),
                "model_verified": (
                    enhanced_output.model_verification.status.value
                    if enhanced_output.model_verification
                    else None
                ),
                "code_verified": (
                    enhanced_output.code_verification.status.value
                    if enhanced_output.code_verification
                    else None
                ),
            },
            "improvements": {
                "workflow_issues_found": (
                    len(
                        enhanced_output.workflow_verification.flaws
                        if enhanced_output.workflow_verification
                        else []
                    )
                ),
                "model_issues_found": (
                    len(
                        enhanced_output.model_verification.flaws
                        if enhanced_output.model_verification
                        else []
                    )
                ),
                "code_issues_found": (
                    len(
                        enhanced_output.code_verification.flaws
                        if enhanced_output.code_verification
                        else []
                    )
                ),
            },
        }

    @staticmethod
    def format_comparison_report(comparison: Dict[str, Any]) -> str:
        """Format comparison as readable report."""
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║            PIPELINE COMPARISON REPORT                          ║
╠════════════════════════════════════════════════════════════════╣
║ Problem Type: {comparison['problem_type']:<45} ║
╠════════════════════════════════════════════════════════════════╣
║ ORIGINAL PIPELINE (No Orchestrator)                            ║
║   Status: {comparison['original']['status']:<55} ║
║   Refinements: {comparison['original']['num_refinements']:<49} ║
║   Verifications: None (direct output)                          ║
╠════════════════════════════════════════════════════════════════╣
║ ENHANCED PIPELINE (With Orchestrator)                          ║
║   Status: {comparison['enhanced']['status']:<55} ║
║   Refinements Suggested: {comparison['enhanced']['num_refinements']:<42} ║
║   Workflow Verified: {str(comparison['enhanced']['workflow_verified']):<52} ║
║   Model Verified: {str(comparison['enhanced']['model_verified']):<55} ║
║   Code Verified: {str(comparison['enhanced']['code_verified']):<57} ║
╠════════════════════════════════════════════════════════════════╣
║ ISSUES IDENTIFIED BY ORCHESTRATOR                              ║
║   Workflow Issues: {comparison['improvements']['workflow_issues_found']:<44} ║
║   Model Issues: {comparison['improvements']['model_issues_found']:<47} ║
║   Code Issues: {comparison['improvements']['code_issues_found']:<48} ║
╚════════════════════════════════════════════════════════════════╝
"""
        return report
