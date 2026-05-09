"""Orchestrator Agent - Verifies and refines outputs from other agents."""

from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import re

from ..core.types import (
    WorkflowVerification,
    ModelVerification,
    CodeVerification,
    VerificationStatus,
    FlawSeverity,
    Flaw,
)


class OrchestratorAgent:
    """
    Verifies and coordinates outputs from Workflow and Model Generators.
    
    Responsibilities:
    1. Workflow Verification: Check if workflow is complete and appropriate
    2. Model Verification: Check if model is mathematically sound and correct
    3. Code Verification: Check if code is syntactically valid and matches math
    4. Coordination: Identify when refinement is needed
    
    From Plan.md Part 9: "The Orchestrator Agent adds a verification and 
    refinement layer to improve robustness."
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4", verbose: bool = False):
        """
        Initialize orchestrator.
        
        Args:
            api_key: OpenAI API key
            model_name: LLM model to use
            verbose: Print detailed output
        """
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0.0,
            model_name=model_name,
            openai_api_key=api_key,
        )
        self.verbose = verbose

    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response."""
        message = HumanMessage(content=prompt)
        response = self.llm.invoke([message])
        return response.content

    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract verification details."""
        # Extract sections from response
        result = {
            "raw_response": response,
            "status": "PASS",
            "flaws": [],
            "confidence": "Medium",
        }

        # Look for status
        if "PASS" in response.upper():
            result["status"] = "PASS"
        elif "CRITICAL" in response.upper():
            result["status"] = "CRITICAL_FLAW"
        elif "NEEDS_REFINEMENT" in response.upper() or "NEEDS REFINEMENT" in response.upper():
            result["status"] = "NEEDS_REFINEMENT"

        # Look for confidence
        if "HIGH" in response.upper():
            result["confidence"] = "High"
        elif "LOW" in response.upper():
            result["confidence"] = "Low"

        return result

    # ==================== WORKFLOW VERIFICATION ====================

    def verify_workflow(
        self,
        workflow: str,
        problem_type: str,
        user_query: str,
    ) -> WorkflowVerification:
        """
        Verify workflow quality and appropriateness.
        
        Checks:
        - Follows chosen template (Type-Tailored or Type-Agnostic)
        - Data retrieval steps appropriate for problem type
        - Workflow components complete
        - Will guide model generation effectively
        
        Args:
            workflow: Generated workflow text
            problem_type: Classified problem type
            user_query: Original user query
            
        Returns:
            WorkflowVerification object with status and identified flaws
        """
        verification_prompt = f"""
You are a Workflow Verification Expert. Review the provided workflow and verify its quality.

PROBLEM TYPE: {problem_type}
USER QUERY: {user_query}

WORKFLOW TO VERIFY:
{workflow}

VERIFICATION CHECKLIST:

Step 1: Template Adherence
- Does the workflow follow a standard optimization workflow template?
- Are there clear sections: Question, Thought, Action, Observation, Final Answer?
- Is the structure appropriate for {problem_type} problems?

Step 2: Data Retrieval Appropriateness  
- Are the CSVQA calls requesting relevant data for the problem?
- Do the data requests match what would be needed for {problem_type}?
- Are there enough retrieval steps or too few?

Step 3: Workflow Completeness
- All workflow components present (Question, Thought, Action, Observation, Answer)?
- Expected output clearly defined?
- Step-by-step guidance present?

Step 4: Will Guide Model Generation
- Is the workflow specific enough to guide the model generation?
- Will an LLM following this workflow produce a complete model?
- Are constraints and objectives clearly outlined?

PROVIDE OUTPUT IN THIS EXACT FORMAT:

STATUS: [PASS / NEEDS_REFINEMENT / CRITICAL_FLAW]

TEMPLATE ADHERENCE: [True/False]
REASON: [one sentence]

DATA RETRIEVAL APPROPRIATE: [True/False]
REASON: [one sentence]

COMPONENTS COMPLETE: [True/False]
REASON: [one sentence]

WILL GUIDE MODEL GEN: [True/False]
REASON: [one sentence]

CONFIDENCE: [High/Medium/Low]

FLAWS IDENTIFIED:
[If none, write "None"]
[If any, list as:
  FLAW 1:
    Severity: [CRITICAL/MAJOR/MINOR]
    Issue: [what is wrong]
    Location: [where in workflow]
    Root Cause: [why it's wrong]
    Fix: [how to fix it]
]

OVERALL NOTES: [2-3 sentences on workflow quality]
"""

        if self.verbose:
            print("[Orchestrator] Verifying workflow...")

        response = self._call_llm(verification_prompt)
        
        # Parse response
        verification = self._parse_workflow_verification(response)
        
        if self.verbose:
            print(f"[Orchestrator] Workflow verification: {verification.status.value}")
        
        return verification

    def _parse_workflow_verification(self, response: str) -> WorkflowVerification:
        """Parse workflow verification response from LLM."""
        lines = response.strip().split("\n")
        
        result = {
            "status": VerificationStatus.PASS,
            "follows_template": True,
            "data_retrieval_appropriate": True,
            "components_complete": True,
            "will_guide_model_generation": True,
            "flaws": [],
            "confidence": "Medium",
            "notes": "",
        }

        # Parse each field
        current_section = None
        flaw_buffer = {}
        
        for line in lines:
            line = line.strip()
            
            if "STATUS:" in line.upper():
                if "CRITICAL" in line.upper():
                    result["status"] = VerificationStatus.CRITICAL_FLAW
                elif "NEEDS_REFINEMENT" in line.upper() or "NEEDS REFINEMENT" in line.upper():
                    result["status"] = VerificationStatus.NEEDS_REFINEMENT
                else:
                    result["status"] = VerificationStatus.PASS
            
            elif "TEMPLATE ADHERENCE:" in line.upper():
                result["follows_template"] = "TRUE" in line.upper()
            
            elif "DATA RETRIEVAL APPROPRIATE:" in line.upper():
                result["data_retrieval_appropriate"] = "TRUE" in line.upper()
            
            elif "COMPONENTS COMPLETE:" in line.upper():
                result["components_complete"] = "TRUE" in line.upper()
            
            elif "WILL GUIDE" in line.upper():
                result["will_guide_model_generation"] = "TRUE" in line.upper()
            
            elif "CONFIDENCE:" in line.upper():
                if "HIGH" in line.upper():
                    result["confidence"] = "High"
                elif "LOW" in line.upper():
                    result["confidence"] = "Low"
                else:
                    result["confidence"] = "Medium"
            
            elif "OVERALL NOTES:" in line.upper():
                current_section = "notes"
            
            elif current_section == "notes" and line:
                result["notes"] += line + " "

        return WorkflowVerification(
            status=result["status"],
            follows_template=result["follows_template"],
            data_retrieval_appropriate=result["data_retrieval_appropriate"],
            components_complete=result["components_complete"],
            will_guide_model_generation=result["will_guide_model_generation"],
            flaws=result["flaws"],
            confidence=result["confidence"],
            notes=result["notes"].strip(),
        )

    # ==================== MODEL VERIFICATION ====================

    def verify_model(
        self,
        model_formulation: str,
        workflow: str,
        user_query: str,
        retrieved_data: Optional[str] = None,
    ) -> ModelVerification:
        """
        Verify model mathematical correctness and problem alignment.
        
        Checks:
        - Consistency with workflow (uses workflow guidance)
        - Completeness (all variables, constraints, objective defined)
        - Mathematical soundness (valid syntax and logic)
        - Problem alignment (solves what user asked)
        - Data integrity (no hallucinated values)
        
        Args:
            model_formulation: Generated mathematical formulation
            workflow: Original workflow that guided generation
            user_query: Original user query
            retrieved_data: Data retrieved from CSVQA (for validation)
            
        Returns:
            ModelVerification object with status and identified flaws
        """
        verification_prompt = f"""
You are a Mathematical Optimization Verification Expert. Review the model formulation.

ORIGINAL QUERY: {user_query}

WORKFLOW PROVIDED TO AGENT:
{workflow}

RETRIEVED DATA:
{retrieved_data if retrieved_data else "None provided"}

MODEL FORMULATION TO VERIFY:
{model_formulation}

VERIFICATION CHECKLIST:

Step 1: Consistency with Workflow
- Does the formulation follow the workflow structure?
- Are data elements from workflow used?
- Did the agent use CSVQA results as indicated?

Step 2: Completeness
- Are all decision variables defined?
- Are all constraints included?
- Is objective function specified?
- Are variable domains/bounds specified?

Step 3: Mathematical Soundness
- Is the formulation mathematically valid?
- Are variable types correct (CONTINUOUS/INTEGER/BINARY)?
- Are constraint logic and bounds sensible?
- Do the mathematical expressions make sense?

Step 4: Problem Alignment
- Does the formulation address user's intent?
- Are all mentioned constraints included?
- Is the objective what user requested?
- Nothing important missing?

Step 5: Data Integrity
- Are all coefficients realistic (not hallucinated)?
- If data was retrieved, is it used correctly?
- Are all referenced data points present?

PROVIDE OUTPUT IN THIS EXACT FORMAT:

STATUS: [PASS / NEEDS_REFINEMENT / CRITICAL_FLAW]

CONSISTENCY WITH WORKFLOW: [True/False]
REASON: [one sentence]

COMPLETENESS: [True/False]
REASON: [one sentence]

MATHEMATICAL SOUNDNESS: [True/False]
REASON: [one sentence]

PROBLEM ALIGNMENT: [True/False]
REASON: [one sentence]

DATA INTEGRITY: [True/False]
REASON: [one sentence]

CONFIDENCE: [High/Medium/Low]

FLAWS IDENTIFIED:
[If none, write "None"]
[If any, list as:
  FLAW 1:
    Severity: [CRITICAL/MAJOR/MINOR]
    Issue: [what is wrong]
    Location: [where in model]
    Root Cause: [why it's wrong]
    Fix: [how to fix it]
]

OVERALL NOTES: [2-3 sentences on model quality]
"""

        if self.verbose:
            print("[Orchestrator] Verifying model formulation...")

        response = self._call_llm(verification_prompt)
        
        verification = self._parse_model_verification(response)
        
        if self.verbose:
            print(f"[Orchestrator] Model verification: {verification.status.value}")
        
        return verification

    def _parse_model_verification(self, response: str) -> ModelVerification:
        """Parse model verification response from LLM."""
        result = {
            "status": VerificationStatus.PASS,
            "consistency_with_workflow": True,
            "completeness": True,
            "mathematical_soundness": True,
            "problem_alignment": True,
            "data_integrity": True,
            "flaws": [],
            "confidence": "Medium",
            "notes": "",
        }

        lines = response.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if "STATUS:" in line.upper():
                if "CRITICAL" in line.upper():
                    result["status"] = VerificationStatus.CRITICAL_FLAW
                elif "NEEDS_REFINEMENT" in line.upper() or "NEEDS REFINEMENT" in line.upper():
                    result["status"] = VerificationStatus.NEEDS_REFINEMENT
                else:
                    result["status"] = VerificationStatus.PASS

            elif "CONSISTENCY" in line.upper():
                result["consistency_with_workflow"] = "TRUE" in line.upper()

            elif "COMPLETENESS:" in line.upper():
                result["completeness"] = "TRUE" in line.upper()

            elif "MATHEMATICAL" in line.upper():
                result["mathematical_soundness"] = "TRUE" in line.upper()

            elif "PROBLEM ALIGNMENT:" in line.upper():
                result["problem_alignment"] = "TRUE" in line.upper()

            elif "DATA INTEGRITY:" in line.upper():
                result["data_integrity"] = "TRUE" in line.upper()

            elif "CONFIDENCE:" in line.upper():
                if "HIGH" in line.upper():
                    result["confidence"] = "High"
                elif "LOW" in line.upper():
                    result["confidence"] = "Low"

            elif "OVERALL NOTES:" in line.upper():
                current_section = "notes"

            elif current_section == "notes" and line:
                result["notes"] += line + " "

        return ModelVerification(
            status=result["status"],
            consistency_with_workflow=result["consistency_with_workflow"],
            completeness=result["completeness"],
            mathematical_soundness=result["mathematical_soundness"],
            problem_alignment=result["problem_alignment"],
            data_integrity=result["data_integrity"],
            flaws=result["flaws"],
            confidence=result["confidence"],
            notes=result["notes"].strip(),
        )

    # ==================== CODE VERIFICATION ====================

    def verify_code(
        self,
        code: str,
        model_formulation: str,
    ) -> CodeVerification:
        """
        Verify code quality and correctness.
        
        Checks:
        - Python syntax valid
        - Gurobi API calls correct
        - Model initialization proper
        - Constraint formulation matches math
        - Data types consistent
        
        Args:
            code: Generated Python code
            model_formulation: Mathematical formulation code should implement
            
        Returns:
            CodeVerification object with status and identified flaws
        """
        verification_prompt = f"""
You are a Python/Gurobi Code Verification Expert. Review the generated code.

MATHEMATICAL FORMULATION:
{model_formulation}

CODE TO VERIFY:
{code}

VERIFICATION CHECKLIST:

Step 1: Syntax Validity
- Is the Python code syntactically correct?
- Are there any obvious syntax errors?

Step 2: Gurobi API Usage
- Are Gurobi API calls correct and valid?
- Is the model initialized properly?
- Are variable types used correctly?

Step 3: Model Initialization
- Is gp.Model created correctly?
- Are variables added with correct types?
- Are variable bounds set correctly?

Step 4: Constraint Formulation
- Do constraints in code match the mathematical formulation?
- Is constraint logic implemented correctly?
- Are variable references correct?

Step 5: Data Type Consistency
- Are all variables consistent in type (no mixing int/continuous)?
- Are objective coefficients compatible with variable types?
- Are constraint bounds appropriate?

PROVIDE OUTPUT IN THIS EXACT FORMAT:

STATUS: [PASS / NEEDS_REFINEMENT / CRITICAL_FLAW]

SYNTAX VALID: [True/False]
REASON: [one sentence]

API CALLS CORRECT: [True/False]
REASON: [one sentence]

MODEL INITIALIZATION PROPER: [True/False]
REASON: [one sentence]

CONSTRAINT FORMULATION MATCHES: [True/False]
REASON: [one sentence]

DATA TYPES CONSISTENT: [True/False]
REASON: [one sentence]

CONFIDENCE: [High/Medium/Low]

FLAWS IDENTIFIED:
[If none, write "None"]
[If any, list as:
  FLAW 1:
    Severity: [CRITICAL/MAJOR/MINOR]
    Issue: [what is wrong]
    Location: [line number or section]
    Root Cause: [why it's wrong]
    Fix: [how to fix it]
]

OVERALL NOTES: [2-3 sentences on code quality]
"""

        if self.verbose:
            print("[Orchestrator] Verifying code...")

        response = self._call_llm(verification_prompt)
        
        verification = self._parse_code_verification(response)
        
        if self.verbose:
            print(f"[Orchestrator] Code verification: {verification.status.value}")
        
        return verification

    def _parse_code_verification(self, response: str) -> CodeVerification:
        """Parse code verification response from LLM."""
        result = {
            "status": VerificationStatus.PASS,
            "syntax_valid": True,
            "api_calls_correct": True,
            "model_initialization_proper": True,
            "constraint_formulation_matches_math": True,
            "data_types_consistent": True,
            "flaws": [],
            "confidence": "Medium",
            "notes": "",
        }

        lines = response.strip().split("\n")
        current_section = None

        for line in lines:
            line = line.strip()

            if "STATUS:" in line.upper():
                if "CRITICAL" in line.upper():
                    result["status"] = VerificationStatus.CRITICAL_FLAW
                elif "NEEDS_REFINEMENT" in line.upper() or "NEEDS REFINEMENT" in line.upper():
                    result["status"] = VerificationStatus.NEEDS_REFINEMENT
                else:
                    result["status"] = VerificationStatus.PASS

            elif "SYNTAX" in line.upper():
                result["syntax_valid"] = "TRUE" in line.upper()

            elif "API CALLS" in line.upper():
                result["api_calls_correct"] = "TRUE" in line.upper()

            elif "MODEL INITIALIZATION" in line.upper():
                result["model_initialization_proper"] = "TRUE" in line.upper()

            elif "CONSTRAINT FORMULATION" in line.upper():
                result["constraint_formulation_matches_math"] = "TRUE" in line.upper()

            elif "DATA TYPES" in line.upper():
                result["data_types_consistent"] = "TRUE" in line.upper()

            elif "CONFIDENCE:" in line.upper():
                if "HIGH" in line.upper():
                    result["confidence"] = "High"
                elif "LOW" in line.upper():
                    result["confidence"] = "Low"

            elif "OVERALL NOTES:" in line.upper():
                current_section = "notes"

            elif current_section == "notes" and line:
                result["notes"] += line + " "

        return CodeVerification(
            status=result["status"],
            syntax_valid=result["syntax_valid"],
            api_calls_correct=result["api_calls_correct"],
            model_initialization_proper=result["model_initialization_proper"],
            constraint_formulation_matches_math=result["constraint_formulation_matches_math"],
            data_types_consistent=result["data_types_consistent"],
            flaws=result["flaws"],
            confidence=result["confidence"],
            notes=result["notes"].strip(),
        )

    # ==================== REFINEMENT RECOMMENDATIONS ====================

    def recommend_refinements(
        self,
        workflow_verification: Optional[WorkflowVerification] = None,
        model_verification: Optional[ModelVerification] = None,
        code_verification: Optional[CodeVerification] = None,
    ) -> Dict[str, Any]:
        """
        Recommend refinement strategy based on verifications.
        
        Returns recommendations for:
        - Workflow replan (if workflow has critical flaws)
        - Model refinement (if model has critical flaws)
        - Code refinement (if code has critical flaws)
        """
        recommendations = {
            "workflow_replan_needed": False,
            "model_refinement_needed": False,
            "code_refinement_needed": False,
            "recommendations": [],
        }

        if workflow_verification:
            if workflow_verification.status == VerificationStatus.CRITICAL_FLAW:
                recommendations["workflow_replan_needed"] = True
                recommendations["recommendations"].append(
                    "CRITICAL: Regenerate workflow due to fundamental flaws"
                )
            elif workflow_verification.status == VerificationStatus.NEEDS_REFINEMENT:
                recommendations["workflow_replan_needed"] = True
                recommendations["recommendations"].append(
                    "Improve workflow based on identified issues"
                )

        if model_verification:
            if model_verification.status == VerificationStatus.CRITICAL_FLAW:
                recommendations["model_refinement_needed"] = True
                recommendations["recommendations"].append(
                    "CRITICAL: Regenerate model formulation due to fundamental flaws"
                )
            elif model_verification.status == VerificationStatus.NEEDS_REFINEMENT:
                recommendations["model_refinement_needed"] = True
                recommendations["recommendations"].append(
                    "Refine model formulation based on identified issues"
                )

        if code_verification:
            if code_verification.status == VerificationStatus.CRITICAL_FLAW:
                recommendations["code_refinement_needed"] = True
                recommendations["recommendations"].append(
                    "CRITICAL: Fix code due to fundamental flaws"
                )
            elif code_verification.status == VerificationStatus.NEEDS_REFINEMENT:
                recommendations["code_refinement_needed"] = True
                recommendations["recommendations"].append(
                    "Fix code issues to make it runnable"
                )

        return recommendations
