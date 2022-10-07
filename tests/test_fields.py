from ftle import fields


class Test_Fields_from_dict:
    def test_is_collection_of_functions(self):
        input_dict = dict(myfield=lambda x, y, z, t: x + 2*y + 3*z + 4*t)
        f = fields.Fields.from_dict(input_dict)
        assert f['myfield'](1, 2, 3, 4) == 1 + 4 + 9 + 16

    def test_can_be_converted_to_dict(self):
        input_dict = dict(myfield=lambda x, y, z, t: x + 2*y + 3*z + 4*t)
        f = fields.Fields.from_dict(input_dict)
        f_dict = dict(f)
        assert list(f_dict.keys()) == ['myfield']
