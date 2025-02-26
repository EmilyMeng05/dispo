#prompt user to type in a number of actions they want to prompt the machine to generate
#the default will be 4
def get_user_input():
    num_actions = 4  # Default value
    try:
        user_input = input(f"How many actions do you want to sample? (current: {num_actions}, press Enter to keep): ")
        if user_input:
            num_actions = int(user_input)
            if num_actions <= 0:
                #ERROR detect
                raise ValueError("Number of actions must be a positive integer.")
    except ValueError:
        print(f"Using current number of actions: {num_actions}")
    return num_actions