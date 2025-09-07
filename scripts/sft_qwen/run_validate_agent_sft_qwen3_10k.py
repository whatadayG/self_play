import os
import json


def validate_formatted_data(file_path):
    print(f"\nValidating formatted data file: {file_path}")

    if not os.path.exists(file_path):
        return False, [f"File does not exist: {file_path}"]

    error_messages = []
    total_examples = 0
    valid_examples = 0

    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                total_examples += 1
                line_errors = []

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    line_errors.append(f"Invalid JSON: {e}")
                    error_messages.append(f"Line {line_num}: {line_errors[-1]}")
                    continue

                if "messages" not in data:
                    line_errors.append("Missing 'messages' field")
                    error_messages.append(f"Line {line_num}: {line_errors[-1]}")
                    continue

                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    line_errors.append("'messages' should be a list with at least 2 messages")
                    error_messages.append(f"Line {line_num}: {line_errors[-1]}")
                    continue

                expected_roles = []

                if messages[0].get("role") != "system":
                    line_errors.append("First message should have role 'system'")
                else:
                    expected_roles.append("system")

                    if len(messages) > 1:
                        second_role = messages[1].get("role")
                        if second_role in ["assistant", "user"]:
                            current_role = second_role
                            expected_roles.append(current_role)

                            for i in range(2, len(messages)):
                                if current_role == "assistant":
                                    current_role = "user"
                                else:
                                    current_role = "assistant"
                                expected_roles.append(current_role)
                        else:
                            line_errors.append(
                                f"Second message should be 'assistant' or 'user', got '{second_role}'"
                            )

                for i, (message, expected_role) in enumerate(zip(messages, expected_roles)):
                    if not isinstance(message, dict):
                        line_errors.append(f"Message {i+1} should be a dictionary")
                        continue

                    if "role" not in message:
                        line_errors.append(f"Message {i+1} missing 'role' field")
                        continue

                    if "content" not in message:
                        line_errors.append(f"Message {i+1} missing 'content' field")
                        continue

                    actual_role = message.get("role")
                    if actual_role != expected_role:
                        line_errors.append(
                            f"Message {i+1} expected role '{expected_role}', got '{actual_role}'"
                        )

                    if not message.get("content", "").strip():
                        line_errors.append(f"Message {i+1} has empty content")

                if not line_errors:
                    valid_examples += 1
                else:
                    for error in line_errors:
                        error_messages.append(f"Line {line_num}: {error}")

    except Exception as e:
        error_messages.append(f"Error reading file: {e}")
        return False, error_messages

    print("Validation Results:")
    print(f"  Total examples: {total_examples}")
    print(f"  Valid examples: {valid_examples}")
    print(f"  Invalid examples: {total_examples - valid_examples}")
    print(
        f"  Success rate: {valid_examples/total_examples*100:.1f}%"
        if total_examples > 0
        else "  Success rate: N/A"
    )

    if error_messages:
        print(f"  Found {len(error_messages)} validation errors:")
        for error in error_messages[:10]:
            print(f"    {error}")
        if len(error_messages) > 10:
            print(f"    ... and {len(error_messages) - 10} more errors")
    else:
        print("  All examples are valid!")

    is_valid = len(error_messages) == 0
    return is_valid, error_messages


def main():
    target = "/home/nickatomlin/georgiazhou/self_play/scripts/sft_qwen/sft_qwen3_10k/agent_total.jsonl"
    valid, errors = validate_formatted_data(target)
    print(f"\nSummary: valid={valid}, errors={len(errors)}")


if __name__ == "__main__":
    main() 