# flashcard_quiz_app.py

flashcards = []

def add_flashcard():
    question = input("Enter the question: ")
    answer = input("Enter the answer: ")
    flashcards.append({'question': question, 'answer': answer})
    print("Flashcard added!\n")

def take_quiz():
    if not flashcards:
        print("No flashcards available. Add some first.\n")
        return

    score = 0
    for i, card in enumerate(flashcards):
        print(f"\nQuestion {i + 1}: {card['question']}")
        user_answer = input("Your answer: ")
        if user_answer.strip().lower() == card['answer'].strip().lower():
            print("Correct!")
            score += 1
        else:
            print(f"Incorrect. The correct answer was: {card['answer']}")
    print(f"\nQuiz complete! Your score: {score}/{len(flashcards)}\n")

def menu():
    while True:
        print("Flashcard Quiz App")
        print("1. Add Flashcard")
        print("2. Take Quiz")
        print("3. Exit")

        choice = input("Enter your choice: ")
        if choice == '1':
            add_flashcard()
        elif choice == '2':
            take_quiz()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.\n")

if __name__ == "__main__":
    menu()
