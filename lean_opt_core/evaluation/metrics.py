"""Evaluation metrics for LEAN-LLM-OPT pipeline."""

from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
import re


class FormulationMatcher:
    """Matches generated formulations against ground truth."""

    @staticmethod
    def extract_formulation_components(formulation: str) -> Dict[str, Any]:
        """
        Extract key components from formulation text.
        
        Returns:
            Dict with keys: objective_type, variables, constraints, objective_expr
        """
        components = {
            "objective_type": None,  # "minimize", "maximize"
            "objective_function": "",
            "decision_variables": [],
            "variable_types": {},  # variable -> type
            "constraints": [],
            "sets": [],
            "parameters": [],
        }

        lines = formulation.strip().split("\n")
        current_section = None

        for line in lines:
            line_lower = line.lower()

            # Detect objective
            if "minimize" in line_lower or "min" in line_lower:
                components["objective_type"] = "minimize"
                current_section = "objective"
            elif "maximize" in line_lower or "max" in line_lower:
                components["objective_type"] = "maximize"
                current_section = "objective"

            # Detect decision variables section
            elif "decision variable" in line_lower or "variable" in line_lower:
                current_section = "variables"

            # Detect constraints section
            elif "constraint" in line_lower or "subject to" in line_lower:
                current_section = "constraints"

            # Detect parameters section
            elif "parameter" in line_lower:
                current_section = "parameters"

            # Detect sets section
            elif "set" in line_lower and "index" in line_lower:
                current_section = "sets"

            # Extract content based on current section
            elif current_section == "objective" and line.strip() and "=" in line:
                components["objective_function"] += line + "\n"

            elif current_section == "variables" and line.strip():
                # Extract variable names
                matches = re.findall(r"([a-zA-Z_][a-zA-Z0-9_\[\]]*)", line)
                for match in matches:
                    if match and match not in ["in", "for", "all"]:
                        components["decision_variables"].append(match)

            elif current_section == "constraints" and line.strip():
                if re.search(r"[<>=]", line):
                    components["constraints"].append(line.strip())

        return components

    @staticmethod
    def calculate_similarity(formulation1: str, formulation2: str) -> float:
        """
        Calculate similarity between two formulations (0-1 scale).
        
        Factors:
        - Same objective type (minimize/maximize): 40%
        - Similar decision variables: 30%
        - Similar constraint structure: 20%
        - Similar parameters: 10%
        
        Args:
            formulation1: Generated formulation
            formulation2: Ground truth formulation
            
        Returns:
            Similarity score 0-1
        """
        comp1 = FormulationMatcher.extract_formulation_components(formulation1)
        comp2 = FormulationMatcher.extract_formulation_components(formulation2)

        score = 0.0

        # Objective type match (40%)
        if comp1["objective_type"] == comp2["objective_type"]:
            score += 0.4
        else:
            score += 0.0

        # Decision variables match (30%)
        vars1_set = set(comp1["decision_variables"])
        vars2_set = set(comp2["decision_variables"])
        if vars1_set and vars2_set:
            intersection = len(vars1_set & vars2_set)
            union = len(vars1_set | vars2_set)
            var_similarity = intersection / union if union > 0 else 0
            score += 0.3 * var_similarity

        # Constraint count similarity (20%)
        const1_count = len(comp1["constraints"])
        const2_count = len(comp2["constraints"])
        if const1_count > 0 and const2_count > 0:
            constraint_similarity = min(const1_count, const2_count) / max(
                const1_count, const2_count
            )
            score += 0.2 * constraint_similarity

        # Parameter similarity (10%)
        params1_set = set(comp1["parameters"])
        params2_set = set(comp2["parameters"])
        if params1_set and params2_set:
            param_similarity = len(params1_set & params2_set) / len(
                params1_set | params2_set
            )
            score += 0.1 * param_similarity

        return min(score, 1.0)

    @staticmethod
    def exact_match(formulation1: str, formulation2: str, tolerance: float = 0.1) -> bool:
        """
        Check if formulations match (within tolerance).
        
        For modeling accuracy (EM), we use similarity >= 0.85 as "match"
        
        Args:
            formulation1: Generated formulation
            formulation2: Ground truth formulation
            tolerance: Similarity threshold for match (0-1)
            
        Returns:
            True if formulations match within tolerance
        """
        similarity = FormulationMatcher.calculate_similarity(formulation1, formulation2)
        return similarity >= (1.0 - tolerance)


class OptimalValueValidator:
    """Validates optimal values from solver output."""

    @staticmethod
    def extract_optimal_value(solver_output: str) -> Optional[float]:
        """
        Extract optimal value from solver output/log.
        
        Looks for patterns like:
        - "Optimal value: 1234.56"
        - "obj: 1234.56"
        - "Objective: 1234.56"
        """
        patterns = [
            r"optimal.*?value\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"obj\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"objective\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            r"solution.*?value\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, solver_output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue

        return None

    @staticmethod
    def compare_optimal_values(
        computed_value: Optional[float],
        ground_truth_value: float,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Compare two optimal values with tolerance.
        
        Args:
            computed_value: Computed optimal value from solver
            ground_truth_value: Known optimal value
            tolerance: Relative tolerance for comparison
            
        Returns:
            True if values match within tolerance
        """
        if computed_value is None:
            return False

        if ground_truth_value == 0:
            return abs(computed_value - ground_truth_value) < tolerance

        relative_error = abs(computed_value - ground_truth_value) / abs(
            ground_truth_value
        )
        return relative_error <= tolerance


class ModelingAccuracy:
    """Computes Modeling Accuracy (EM) metrics."""

    @staticmethod
    def compute_em(
        generated_formulations: List[str],
        ground_truth_formulations: List[str],
        similarity_threshold: float = 0.85,
    ) -> Tuple[float, Dict[str, int]]:
        """
        Compute Modeling Accuracy (EM).
        
        EM = (# formulations matching ground truth) / (# total problems)
        
        Match criteria: Similarity >= threshold
        
        Args:
            generated_formulations: List of generated formulations
            ground_truth_formulations: List of ground truth formulations
            similarity_threshold: Threshold for match (0-1)
            
        Returns:
            Tuple of (EM percentage, detailed statistics dict)
        """
        if len(generated_formulations) != len(ground_truth_formulations):
            raise ValueError("Formulation lists must have same length")

        if not generated_formulations:
            return 0.0, {"total": 0, "matches": 0}

        matches = 0
        similarities = []

        for gen, truth in zip(generated_formulations, ground_truth_formulations):
            similarity = FormulationMatcher.calculate_similarity(gen, truth)
            similarities.append(similarity)

            if similarity >= similarity_threshold:
                matches += 1

        em_percentage = (matches / len(generated_formulations)) * 100

        stats = {
            "total": len(generated_formulations),
            "matches": matches,
            "em_percentage": em_percentage,
            "mean_similarity": sum(similarities) / len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
        }

        return em_percentage, stats

    @staticmethod
    def compute_optimal_accuracy(
        computed_values: List[Optional[float]],
        ground_truth_values: List[float],
        tolerance: float = 1e-4,
    ) -> Tuple[float, Dict[str, int]]:
        """
        Compute Optimal Value Accuracy.
        
        Accuracy = (# problems where optimal matches) / (# total problems)
        
        Args:
            computed_values: List of computed optimal values
            ground_truth_values: List of ground truth optimal values
            tolerance: Relative tolerance for comparison
            
        Returns:
            Tuple of (accuracy percentage, statistics dict)
        """
        if len(computed_values) != len(ground_truth_values):
            raise ValueError("Value lists must have same length")

        if not computed_values:
            return 0.0, {"total": 0, "matches": 0}

        matches = 0

        for computed, truth in zip(computed_values, ground_truth_values):
            if OptimalValueValidator.compare_optimal_values(
                computed, truth, tolerance
            ):
                matches += 1

        accuracy_percentage = (matches / len(computed_values)) * 100

        stats = {
            "total": len(computed_values),
            "matches": matches,
            "accuracy_percentage": accuracy_percentage,
        }

        return accuracy_percentage, stats


class TokenCounter:
    """Estimates token usage for API calls."""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count using simple heuristic.
        
        Rule of thumb: ~4 characters per token for English text
        """
        return max(1, len(text) // 4)

    @staticmethod
    def estimate_pipeline_tokens(
        user_query: str,
        refdata_retrieved: str,
        workflow: str,
        csvdata_retrieved: str,
        formulation: str,
    ) -> Dict[str, int]:
        """
        Estimate total tokens for complete pipeline execution.
        
        Includes:
        - Classification agent (query + refdata)
        - Workflow generation (query + refdata + workflow)
        - Model generation (workflow + csvdata)
        """
        tokens = {
            "classification_input": TokenCounter.estimate_tokens(
                user_query + refdata_retrieved
            ),
            "classification_output": TokenCounter.estimate_tokens(workflow),
            "model_gen_input": TokenCounter.estimate_tokens(
                workflow + csvdata_retrieved
            ),
            "model_gen_output": TokenCounter.estimate_tokens(formulation),
        }

        tokens["total"] = sum(tokens.values())

        return tokens
