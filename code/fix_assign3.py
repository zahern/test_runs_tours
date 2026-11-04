with open('Z:/test_runs_tours/code/assign_sa1_to_population.py', 'r') as f:
    content = f.read()

# Find the exact text block
idx = content.find('Check if SA1 EMME crosswalk')
if idx >= 0:
    # Find the end of the block (next double newline after the else block)
    end_idx = content.find('\n\n    # Load 2016', idx)
    if end_idx >= 0:
        old = content[idx:end_idx]
        print('Found block:')
        print(repr(old))
        
        new = '''# Do NOT restrict to study-area SA1s - assign HOME_SA1 to ALL persons.
    # stage5 will fall back to SA2-averaged skims for SA1s outside STM boundary.
    # if os.path.exists(SA1_EMME_XW):
    #     xwalk = pd.read_csv(SA1_EMME_XW, dtype={"SA1_CODE21": str})
    #     study_sa1s = set(xwalk["SA1_CODE21"].str.strip())
    #     sa1_lu = sa1_lu[sa1_lu["SA1_CODE21"].isin(study_sa1s)]
    #     print(f"Restricted to {len(sa1_lu)} study-area SA1 codes")
    # else:
    #     print(f"WARNING: {SA1_EMME_XW} not found -- including all SA1 codes for each SA2")
    print(f"Using all {len(sa1_lu)} SA1 codes from lookup table (no study-area restriction)")'''
        
        content = content[:idx] + new + content[end_idx:]
        with open('Z:/test_runs_tours/code/assign_sa1_to_population.py', 'w') as f:
            f.write(content)
        print('Replaced successfully')
    else:
        print('Could not find end of block')
else:
    print('Start not found')