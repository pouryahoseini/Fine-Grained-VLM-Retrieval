import json
from xml.parsers.expat import model
import ollama
from pydantic import BaseModel
from typing import List, Tuple
from tqdm import tqdm
from utils import parse_stanford_mat_files

# Strict schema the LLM must adhere to
class VehicleCaptions(BaseModel):
    captions: List[str]

# Schema for the LLM judge to evaluate caption validity
class CaptionEvaluation(BaseModel):
    is_valid: bool
    reason: str

class OllamaCaptionGenerator:
    """Generates diverse captions for each vehicle class using a local LLM (Ollama) and saves them to a JSON file.
    The generated captions are designed to be used for data augmentation during training, 
    and the JSON file is loaded by the dataset class to provide varied textual descriptions for each image.
    Args:
        model_name (str): The name of the Ollama model to use for caption generation.
        output_file (str): The path to the JSON file where the generated captions will be saved.
    """

    def __init__(self, model_name: str = "qwen2.5:14b", output_file: str = "llm_captions.json"):
        self.model_name = model_name
        self.output_file = output_file

    def _parse_label(self, label_name: str) -> Tuple[str, str, str]:
        """Parses the label name to extract the year, make, and model for prompt generation.
        Args:
            label_name (str): The name of the vehicle class, e.g., "2010 Ford F-150 Pickup".
        Returns:
            Tuple[str, str, str]: A tuple containing the year, make, and model.
        """

        parts = label_name.split(" ")
        year = parts[-1]
        make = parts[0]
        model_and_type = " ".join(parts[1:-1])

        return year, make, model_and_type

    def generate_prompt(self, label_name: str) -> str:
        """Crafts a detailed prompt for the LLM to generate diverse captions for a given vehicle class, 
        while enforcing strict rules to prevent hallucination of visual details that cannot be inferred from the label alone.
        Args:
            label_name (str): The name of the vehicle class.
        Returns:
            str: The generated prompt.
        """

        year, make, model = self._parse_label(label_name)

        output_prompt = f"""
        You are an expert synthetic data generator for a law enforcement vehicle retrieval system.
        The target vehicle class is: {year} {make} {model}.

        Your task is to generate 8 diverse search queries an investigator might type to find this vehicle.

        CRITICAL RULES:
        1. DO NOT invent colors, backgrounds, or aftermarket parts (no flatbeds, no roof racks).
        2. DO NOT use conversational filler (e.g., "This is a...", "Note the..."). 
        3. Vary the specificity: drop the year sometimes, use common abbreviations (e.g., "Chevy" for "Chevrolet"), or just use the make and body type.

        EXAMPLE INPUT: 2010 Ford F-150 Pickup
        EXAMPLE OUTPUT: 
        [
            "2010 Ford F-150 Pickup",
            "Ford F150 truck",
            "2010 Ford pickup",
            "F-150",
            "older Ford F-150 model",
            "Ford F-150 Pickup from 2010",
            "Ford pickup truck",
            "2010 F150"
        ]

        Generate the JSON array for the target vehicle: {year} {make} {model}.
        """

        return output_prompt, year, make, model

    def call_ollama(self, prompt: str) -> List[str]:
        """Calls Ollama and mathematically forces the output to match the Pydantic schema.
        Args:
            prompt (str): The prompt to send to the Ollama model.
        Returns:
            List[str]: A list of generated captions."""
        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            format=VehicleCaptions.model_json_schema(), # Force guided generation
            options={'temperature': 0.4}
        )
        
        # Get LLM response as a raw JSON string
        raw_json_string = response['message']['content']
        
        # Parse it using Pydantic to ensure it matches the expected schema, and extract the captions
        validated_data = VehicleCaptions.model_validate_json(raw_json_string)
        return validated_data.captions
    
    def llm_guardrail(self, caption: str, car_year: str, car_make: str, car_model: str, llm_model: str = "qwen2.5:14b") -> bool:
        """Uses a local LLM to semantically judge the generated caption.
        Args:
            caption (str): The generated caption to evaluate.
            car_year (str): The year of the vehicle (e.g., "2010").
            car_make (str): The make of the vehicle (e.g., "Ford").
            car_model (str): The model of the vehicle (e.g., "F-150").
        Returns:
            bool: True if the caption is valid, False otherwise.
        """

        judge_prompt = f"""
        You are a strict data quality inspector.
        Evaluate this search query for a vehicle: "{caption}"
        Target Vehicle: {car_year} {car_make} {car_model}

        It is INVALID if it violates ANY of these rules:
        1. It contains ANY mention of color (e.g., crimson, dark, light, silver, red) or location (e.g., "in the city", "on the highway").
        2. It mentions a different car brand or model or year than the target.
        3. It contains conversational questions or filler (e.g., "Where can I find...").
        4. It describes aftermarket modifications that cannot be inferred from the label (e.g., "with roof rack", "flatbed").
        5. It is ok to be less specific than the target (e.g., just "Ford F-150" or "2010 pickup truck" is fine), 
        but it cannot be more specific by adding details not in the label.
        6. DO NOT make up any visual details that cannot be directly inferred from the label.
        
        Remember, the goal is to prevent any hallucinated details that could cause the model to overfit to synthetic patterns during training,
        while sometimes allowing for less specificity to encourage diversity. Judge strictly but fairly.
        
        Is this caption valid for training?
        """
        
        response = ollama.chat(
            model=llm_model,
            messages=[{'role': 'user', 'content': judge_prompt}],
            format=CaptionEvaluation.model_json_schema(),
            options={'temperature': 0.0} # Deterministic judging
        )
        
        evaluation = CaptionEvaluation.model_validate_json(response['message']['content'])
        
        if not evaluation.is_valid:
            print(f"Rejected: '{caption}' | Reason: {evaluation.reason}")
            
        return evaluation.is_valid

    def build_dataset(self, unique_labels: List[str]):
        """Generates captions for each unique vehicle class and saves them to a JSON file.
        Args:
            unique_labels (List[str]): A list of unique vehicle class names extracted from the dataset.
        """

        captions_dict = {}
        
        print(f"Generating captions for {len(unique_labels)} unique classes using {self.model_name}...")
        
        for label in tqdm(unique_labels):
            prompt, year, make, model_and_type = self.generate_prompt(label)

            try:
                # Call the LLM to generate captions based on the crafted prompt
                captions_dict[label] = self.call_ollama(prompt)

                # Apply the guardrail to filter out any captions that violate the rules
                safe_captions = [c for c in captions_dict[label] if self.llm_guardrail(c, year, make, model_and_type, self.model_name)]

                # If the LLM completely failed and returned 0 safe captions, use the fallback
                if len(safe_captions) == 0:
                    raise ValueError(f"All generated captions were rejected by the guardrail for label {label}.")
                
                captions_dict[label] = safe_captions
            except Exception as e:
                print(f"\nFailed on {label}: {e}")

                # Fallback to a simple caption if the LLM call fails for any reason
                captions_dict[label] = [f"A photo of a {label}."]

        # Save the generated captions to a JSON file for later use in the dataset
        with open(self.output_file, "w") as f:
            json.dump(captions_dict, f, indent=4)
        print(f"\nSuccessfully saved to {self.output_file}")


def generate_captions_for_dataset(data_dir: str, output_file: str = "llm_captions.json", model_name: str = "qwen2.5:14b"):
    """Main function to generate captions for the dataset.
    Args:
        data_dir (str): The directory containing the Stanford Cars dataset.
        output_file (str): The path to the JSON file where the generated captions will be saved.
        model_name (str): The name of the Ollama model to use for caption generation.
    """

    print(f"Parsing dataset at {data_dir} to extract class names...")
    _, train_class_names = parse_stanford_mat_files(data_dir, split="train")
    _, test_class_names = parse_stanford_mat_files(data_dir, split="test")
    
    # Combine unique class names from both splits
    class_names = list(set(train_class_names + test_class_names))
    
    print(f"Found {len(class_names)} unique classes.")
    
    generator = OllamaCaptionGenerator(model_name=model_name, output_file=output_file)
    generator.build_dataset(class_names)


if __name__ == "__main__":
    generate_captions_for_dataset(data_dir="./data/stanford_cars", output_file="./data/llm_captions.json", model_name="qwen2.5:14b")