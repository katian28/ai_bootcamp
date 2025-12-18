from generate import GenerateEmail

# Use your GPT-4.1 deployment directly
generator = GenerateEmail(model="gpt-4.1")

# Test with a simple email
test_email = "Hi, just checking in on the meeting notes."

print("Testing API call...")
result = generator.generate("shorten", test_email)

if result:
    print(f"✓ API Works!\nResult: {result}")
else:
    print("✗ API call failed")
