def threeToOne(variant):
    if variant == "ALA":
        variant = "A"
    elif variant == "ARG":
        variant = "R"
    elif variant == "VAL":
        variant = "V"
    elif variant == "GLU":
        variant = "E"
    elif variant == "PRO":
        variant = "P"
    elif variant == "LEU":
        variant = "L"
    elif variant == "GLY":
        variant = "G"
    elif variant == "ASN":
        variant = "N"
    elif variant == "SER":
        variant = "S"
    elif variant == "GLN":
        variant = "Q"
    elif variant == "THR":
        variant = "T"
    elif variant == "MET":
        variant = "M"
    elif variant == "LYS":
        variant = "K"
    elif variant == "ASP":
        variant = "D"
    elif variant == "ILE":
        variant = "I"
    elif variant == "PHE":
        variant = "F"
    elif variant == "TRP":
        variant = "W"
    elif variant == "TYR":
        variant = "Y"
    elif variant == "HIS":
        variant = "H"
    elif variant == "CYS":
        variant = "C"
    elif variant == 'UNK':
        variant = 'X'
    elif variant == 'ASX':
        variant = 'O'
    return (variant)



def convert_non_standard_amino_acids(sequence):
    """
    Convert non-standard or ambiguous amino acid codes to their closest relatives.
    """

    # Define a dictionary to map non-standard codes to standard amino acids
    conversion_dict = {
        'B': 'D',  # Aspartic Acid (D) is often used for B (Asx)
        'Z': 'E',  # Glutamic Acid (E) is often used for Z (Glx)
        'X': 'A',  # Alanine (A) is a common placeholder for unknown/ambiguous
        'U': 'C',  # Cysteine (C) is often used for Selenocysteine (U)
        'J': 'L',  # Leucine (L) is often used for J (Leu/Ile)
        'O': 'K',  # Lysine (K) is often used for O (Pyrrolysine)
        # '*' or 'Stop' represents a stop codon; you may replace with '' to remove
        '*': '',
    }

    # Replace non-standard codes with their closest relatives
    converted_sequence = ''.join([conversion_dict.get(aa, aa) for aa in sequence])

    return converted_sequence
