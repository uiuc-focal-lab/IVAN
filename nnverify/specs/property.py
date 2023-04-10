from nnverify.specs.input_spec import InputSpecType, InputSpec
from nnverify.specs.out_spec import OutSpecType


class Property:
    def __init__(self, input_lbs, input_ubs, inp_type, out_constr, dataset, input=None):
        if inp_type == InputSpecType.LINF:
            self.input_specs = [InputSpec(input_lbs, input_ubs, out_constr, dataset, input=input)]
        # Since the properties in this case can be conjunctive
        elif inp_type == InputSpecType.PATCH:
            self.input_specs = []
            for i in range(len(input_lbs)):
                self.input_specs.append(InputSpec(input_lbs[i], input_ubs[i], out_constr, dataset, input=input))
        elif inp_type == InputSpecType.GLOBAL:
            # A property may contain multiple clauses
            self.input_specs = []
            for i in range(len(input_lbs)):
                self.input_specs.append(InputSpec(input_lbs[i], input_ubs[i], out_constr[i], dataset))
        else:
            raise ValueError("Unsupported Input property type!")

        self.inp_type = inp_type
        self.out_constr = out_constr
        self.dataset = dataset

    def is_local_robustness(self):
        return self.out_constr.constr_type == OutSpecType.LOCAL_ROBUST

    def get_label(self):
        if self.out_constr.constr_type is not OutSpecType.LOCAL_ROBUST:
            raise ValueError("Label only for local robustness properties!")
        return self.out_constr.label

    def get_input_clause_count(self):
        return len(self.input_specs)

    def get_input_clause(self, i):
        return self.input_specs[i]
