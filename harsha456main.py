import random

def coin_toss():
    return random.choice(["Heads", "Tails"])

def multiple_tosses(num_flips):
    heads_count = 0
    tails_count = 0
    
    for _ in range(num_flips):
        result = coin_toss()
        print(result)
        if result == "Heads":
            heads_count += 1
        else:
            tails_count += 1
    
    print("\nResults:")
    print(f"Heads: {heads_count} ({(heads_count/num_flips)*100:.2f}%)")
    print(f"Tails: {tails_count} ({(tails_count/num_flips)*100:.2f}%)")

def main():
    while True:
        try:
            num_flips = input("Enter the number of coin flips (or press Enter for default 10): ").strip()
            num_flips = int(num_flips) if num_flips else 10  # Default to 10 if input is empty
            if num_flips <= 0:
                print("Please enter a positive number.")
                continue
            multiple_tosses(num_flips)
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        
        again = input("Do you want to flip again? (yes/no, default: no): ").strip().lower()
        if again not in ('yes', 'y'):
            print("Thank you for playing!")
            break

if __name__ == "__main__":
    try:
        main()
    except OSError:
        print("I/O error detected. Running in non-interactive mode with 10 flips.")
        multiple_tosses(10)
