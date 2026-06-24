import json
import os
from datetime import datetime
import argparse

# Error codes
errors = {"MISSING_FIELD": 0,
          "MISMATCHED_LENGTH": 0,
          "EMPTY_TURN": 0}



def parse_args():
    """
    Parse command line arguments for dataset processing.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process and format dataset for streaming turns.")
    parser.add_argument('--user_tag', type=str, default="user", help='Key for user turn in the data')
    parser.add_argument('--thoughts_tag', type=str, default="thoughts", help='Key for thoughts/summary in the data')
    parser.add_argument('--responder_tag', type=str, default="responder", help='Key for responder turn in the data')
    parser.add_argument('--base_data_dir', type=str, default="../generated_data", help='Directory containing input data files')
    parser.add_argument('--output_path', type=str, default="formatted_data.jsonl", help='Output file path')
    parser.add_argument('--include_history', action='store_true', help='Include previous responses in output (use generate_turn_dict)')
    return parser.parse_args()

args = parse_args()
user_tag = args.user_tag
thoughts_tag = args.thoughts_tag
responder_tag = args.responder_tag
base_data_dir = args.base_data_dir
output_path = args.output_path
include_history = args.include_history

# Unrolls the single turn into a streaming input
def generate_turn_dict(turn, turn_id, previous_user_turn, previous_responder_turn):
    user_turn = turn[user_tag]
    # Lists
    thought_turn = turn[thoughts_tag]
    responder_turn = turn[responder_tag]

    streaming_data = []

    previous_responses = ""
    current_thought = ""
    current_response = ""

    num_phrases = len(thought_turn)

    for i in range(num_phrases):      
        if i != 0:
            previous_responses += f" {responder_turn[i-1]}"
        current_thought = thought_turn[i]
        current_response = responder_turn[i]

        streaming_data.append({
            "previous_responses": previous_responses,
            "current_thought": current_thought,
            "current_response": current_response

        })

    turn_dict = {
        "turn_id": turn_id,
        "user": user_turn,
        "grouped_responses": streaming_data
    }

    return turn_dict, user_turn, responder_turn

def generate_turn_dict_no_history(turn, turn_id, previous_user_turn, previous_responder_turn):
    user_turn = turn[user_tag]
    # Lists
    thought_turn = turn[thoughts_tag]
    responder_turn = turn[responder_tag]

    num_phrases = len(thought_turn)
    
    streaming_data = []

    for i, _ in enumerate(range(num_phrases)):
        streaming_data.append({
            "current_response": responder_turn[i],
            "current_thought": thought_turn[i]
        })

    turn_dict = {
        "turn_id": turn_id,
        "user": user_turn,
        "grouped_responses": streaming_data
    }

    return turn_dict, user_turn, responder_turn


#-----------------------------------------#
# Check if the data generated is valid (i.e. matching streams)
# DO NOT MODIFY
#-----------------------------------------#
def invalid_turn(turn):
    if 'user' in turn:
        p1_turn = turn[user_tag]
    else:
        errors["MISSING_FIELD"] += 1
        return True
    if 'thoughts' in turn:
        p2_stream = turn[thoughts_tag]
    else:
        errors["MISSING_FIELD"] += 1
        return True
    if 'response' in turn:
        summarized_p2 = turn[responder_tag]
    else:
        errors["MISSING_FIELD"] += 1
        return True
    
    mismatched_length = (len(summarized_p2) != len(p2_stream))
    if mismatched_length:
        errors["MISMATCHED_LENGTH"] += 1
    empty_p2 = (len(summarized_p2) == 0) or (len(p2_stream) == 0) 
    if empty_p2: 
        errors["EMPTY_TURN"] += 1
    return empty_p2 or mismatched_length

global_conv_id = 0 

def process_and_save_grouped(input_path, output_path, global_conv_id): 
    """
    Processes a data file, validates turns, formats them, and saves to output.
    Args:
        input_path (str): Path to the input file.
        output_path (str): Path to the output file.
        global_conv_id (int): Global conversation ID counter.
    Returns:
        int: Updated global conversation ID.
    """
    print_example = True
    invalid_turn_count = 0
    with open(input_path, "r") as infile:
        for i, line in enumerate(infile):
            try:
                turns = json.loads(line)["conversation"]
            except:
                print(i, line, input_path)
                return global_conv_id

            invalid_turn_check = False

            for n, turn in enumerate(turns):
                if (invalid_turn(turn)):
                    invalid_turn_check = True
                    print(f"TURN INVALID AT {n}")
                    break
            
            if (invalid_turn_check):
                invalid_turn_count += 1
                continue

            previous_user_turn = ""
            previous_responder_turn = ""
            grouped_turns = []
            for turn_id, turn in enumerate(turns):
                if include_history:
                    turn_dict, next_prev_user_turn, next_prev_responder_turn = generate_turn_dict(turn, turn_id, previous_user_turn, previous_responder_turn)
                else:
                    turn_dict, next_prev_user_turn, next_prev_responder_turn = generate_turn_dict_no_history(turn, turn_id, previous_user_turn, previous_responder_turn)
                
                last_previous_responder_phrase = ""
                turn_dict["conv_id"] = global_conv_id
                turn_dict["previous_user_turn"] = previous_user_turn
                turn_dict["previous_responder_turn"] = previous_responder_turn
                turn_dict["last_previous_responder_phrase"] = last_previous_responder_phrase



                previous_user_turn = next_prev_user_turn
                previous_responder_turn = next_prev_responder_turn


                grouped_turns.append(turn_dict)

            if None in grouped_turns:
                print("Grouped turns failure, correct the error and run again")
                continue
            
            with open(output_path, "a") as outfile:
                for turn in grouped_turns:
                    outfile.write(json.dumps(turn) + "\n")
                if print_example:
                    print_example = False
            global_conv_id += 1
            print(f"Global conv id: {global_conv_id}")
            
    return global_conv_id


def main():
    """
    Main function to process all files in the base data directory.
    """
    global global_conv_id

    with open(output_path, "w") as f:
        # Ensure the output file is created and empty before processing 
        print(f"Output file created at {output_path}") 

    for conv_file in os.listdir(f"{base_data_dir}"):
        global_conv_id = process_and_save_grouped(
            f"{base_data_dir}/{conv_file}",
            output_path,
            global_conv_id
        )
    pass

if __name__ == "__main__":
    main()