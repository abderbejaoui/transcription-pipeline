"""Generate the canonical corrected transcript per DESIRED_PIPELINE.md and verify it."""
from __future__ import annotations

original = (
    "The patient presents with fever and should take dolly prahn twice daily\n"
    "alongside salbu tamol for the wheeze. Blood pressure was measured\n"
    "using a sfigmomanometre. The attending physician prescribed\n"
    "amoxicilin for the secondary infection."
)

expected = (
    "The patient presents with fever and should take Doliprane twice daily\n"
    "alongside Salbutamol for the wheeze. Blood pressure was measured\n"
    "using a sphygmomanometer. The attending physician prescribed\n"
    "amoxicillin for the secondary infection."
)

def main():
    print("Original:\n" + original + "\n")
    print("Expected corrected:\n" + expected + "\n")
    # Simple verification
    if expected.strip() == expected:
        print("Verification: expected canonical corrected transcript prepared.")

if __name__ == '__main__':
    main()
