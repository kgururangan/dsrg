import re

def convert_name(name, string, nbody):

    num_alpha = len(re.findall(r'[a-z]', string)) // 2
    num_beta = len(re.findall(r'[A-Z]', string)) // 2
    spincase = 'a' * num_alpha + 'b' * num_beta

    if 'H' in name:
        new_name = 'h' + f"['{spincase}']"
    elif 'C' in name:
        new_name = 'O' + f"['{spincase}']"
    elif 'T' in name:
        new_name = 't' + f"['{spincase}']"
    elif 'Gamma' in name:
        new_name = 'gamma1' + f"['{spincase}']"
    elif 'Eta' in name:
        new_name = 'eta1' + f"['{spincase}']"
    elif 'Lambda' in name:
        new_name = 'lambdas' + f"['{spincase}']"
    else:
        new_name = name
    return new_name

def convert_op(op):
    name, einsumstr, _ = op.split("\"")
    name = name[:-1]
    nbody = len(einsumstr) // 2

    einsumstr = einsumstr[nbody:] + einsumstr[:nbody]
    name = convert_name(name, einsumstr, nbody)
    return name, einsumstr

def main():

    with open('xxx', 'r') as f:
        for line in f.readlines():
            if "::" in line or '//' in line: continue
            l = line.split()
            if l:
                ops = []
                ops.append(l[0])
                sign = l[1]
                ops = []
                for i in l[2:]:
                    if "[" in i:
                        ops.append(i)
                
                for op in ops:
                    name, einsumstr = convert_op(op)
                    print(name, einsumstr)
                    
                




                # alpha = l[2]
                # * = l[3]
                #try:
                #    print(l[4])
                #    weight = float(l[4])
                #    start = 5
                #except ValueError:
                #    weight = 1.0
                #    start = 4
                #rhs = []
                #for i in l[start:]:
                #    if "*" not in i:
                #        rhs.append(i)
                #print(lhs, sign, weight, rhs)


if __name__ == "__main__":
    main()
