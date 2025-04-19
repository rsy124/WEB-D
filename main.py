from fact_verification import verify_fact

# Take user input
claim = input("Enter a claim to verify: ")

# Call the verification function
truth_score, explanation, sources = verify_fact(claim)

# Print the result
print("\nVerification Result:")
print("Truth Score:", truth_score)
print("Explanation:", explanation)
print("Sources Used:", ", ".join(sources) if sources else "No sources found.")