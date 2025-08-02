sis = ss.SIS(
    beta = dict(
        random = ss.permonth([0.005, 0.001]),
        prenatal = [ss.permonth(0.1), 0],
        postnatal = [ss.permonth(0.1), 0]
    )
) 