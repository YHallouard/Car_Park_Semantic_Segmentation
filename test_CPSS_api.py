import unittest
from app import app

class TestStringMethods(unittest.TestCase):

    def test_index_route_should_return_hello_world(self):
        # GIVEN
        tester = app.test_client(self)

        expected_res = b'{"hello":"world"}\n'

        # WHEN
        res = tester.get('/')
        res = res.get_data()

        # THEN
        self.assertEqual(expected_res, res)