from __future__ import annotations

import unittest

from PySide6.QtCore import QCoreApplication

from eve_sim.fleet_setup import EftFitParser
from eve_sim.user_errors import UserFacingError, display_user_error


class LocalizationTests(unittest.TestCase):
    def test_display_user_error_translates_structured_user_error(self) -> None:
        error = UserFacingError(
            "Ship not found in pyfa: {name}",
            name="Ferox",
        )

        self.assertEqual(
            display_user_error(error),
            QCoreApplication.translate("eve_sim", 'Ship not found in pyfa: {name}').format(name="Ferox"),
        )

    def test_display_user_error_keeps_raw_error_text(self) -> None:
        raw_error = "pyfa module not found: Warp Scrambler II"
        self.assertEqual(
            display_user_error(raw_error),
            raw_error,
        )

    def test_eft_parser_empty_text_raises_structured_error(self) -> None:
        with self.assertRaises(UserFacingError) as ctx:
            EftFitParser().parse("")

        self.assertEqual(ctx.exception.source_text, "Fit text is empty.")
        self.assertEqual(
            display_user_error(ctx.exception),
            QCoreApplication.translate("eve_sim", 'Fit text is empty.'),
        )

    def test_eft_parser_invalid_header_raises_structured_error(self) -> None:
        with self.assertRaises(UserFacingError) as ctx:
            EftFitParser().parse("Ferox, Rail DPS")

        self.assertEqual(ctx.exception.source_text, "EFT header is invalid; expected [Ship, Fit Name].")
        self.assertEqual(
            display_user_error(ctx.exception),
            QCoreApplication.translate("eve_sim", 'EFT header is invalid; expected [Ship, Fit Name].'),
        )


if __name__ == "__main__":
    unittest.main()
