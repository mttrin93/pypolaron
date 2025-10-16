import numpy as np
from typing import Dict, Any

kB_eV = 8.617333262145e-5  # Boltzmann constant in eV/K


class PolaronThermodynamics:

    # TODO: this functions can be used to filter out data based on their importance for the temperatures we
    # are interested in, for instance if some polaron becomes satble at 2000 K, but we are not interested at
    # that temperature
    def compute_boltzmann_probabilities(
        self, multi_polaron_results: Dict[str, Any], temperature: float
    ) -> Dict[str, Any]:
        """
        Compute Boltzmann probabilities for multi-polaron configurations from
         polaron formation energies.

        Args:
            multi_polaron_results: output of analyze_multi_polaron()
            temperature: temperature in Kelvin

        Returns:
            Same structure as input but adds 'probability' to each configuration
        """
        all_formation_energies = []
        # Sum formation energies per combination
        for item in multi_polaron_results["multi_polaron_results"]:
            E_total = sum([r["formation_energy"] for r in item["results"]])
            all_formation_energies.append(E_total)

        # Convert to Boltzmann weights
        all_formation_energies = np.array(all_formation_energies)
        weights = np.exp(-all_formation_energies / (kB_eV * temperature))
        probabilities = weights / np.sum(weights)

        # Assign probabilities back
        for idx, item in enumerate(multi_polaron_results["multi_polaron_results"]):
            item["total_formation_energy"] = all_formation_energies[idx]
            item["probability"] = probabilities[idx]

        return multi_polaron_results

    def compute_site_occupations(
        self, multi_polaron_results: Dict[str, Any], atomic_count: int
    ) -> np.ndarray:
        """
        Compute average polaron site occupations from multi-polaron Boltzmann probabilities.

        Args:
            multi_polaron_results: output from compute_boltzmann_probabilities()
            atomic_count: total number of atoms in the structure

        Returns:
            site_occupations: array of length atomic_count with occupation probability per atom
        """
        site_occupations = np.zeros(atomic_count)

        for combo_item in multi_polaron_results["multi_polaron_results"]:
            prob = combo_item["probability"]
            for r in combo_item["results"]:
                atom_idx = r["polaron_atom"]
                site_occupations[
                    atom_idx
                ] += prob  # weighted by configuration probability

        # Ensure occupation does not exceed 1 for single polarons per site
        site_occupations = np.clip(site_occupations, 0, 1)
        return site_occupations
