# Write an epidemics model to JSON, for ss.import_from() in Starsim
#
#     Rscript dump_epidemics.R model.R model.json    # an R script that calls model_default(), etc.
#
# Needs only epidemics and jsonlite. The script is run with model_default(), model_vacamole(),
# model_diphtheria(), and model_ebola() replaced by functions that record their arguments, so nothing
# is simulated; each call is written as a run (several calls, e.g. one per scenario, are several runs).
# Statements that fail are skipped with a message, and the rest of the script still runs.
suppressMessages(library(epidemics))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript dump_epidemics.R <model.R> <out.json>")
infile <- normalizePath(args[1])
outfile <- file.path(normalizePath(dirname(args[2])), basename(args[2]))
fake <- new.env()
fake$.runs <- list()

describe_population <- function(p) {
  cm <- unclass(p$contact_matrix)
  ic <- unclass(p$initial_conditions)
  list(
    name = p$name,
    groups = if (!is.null(rownames(cm))) rownames(cm) else names(p$demography_vector),
    contact_matrix = unname(lapply(seq_len(nrow(cm)), function(i) unname(cm[i, ]))),
    demography_vector = unname(as.numeric(p$demography_vector)),
    compartments = colnames(ic),
    initial_conditions = unname(lapply(seq_len(nrow(ic)), function(i) unname(ic[i, ])))
  )
}

describe_intervention <- function(x) {
  list(
    name = x$name,
    type = if (is_contacts_intervention(x)) "contacts" else "rate",
    time_begin = as.numeric(x$time_begin),
    time_end = as.numeric(x$time_end),
    reduction = unname(lapply(seq_len(NROW(x$reduction)), function(i) as.numeric(as.matrix(x$reduction)[i, ])))
  )
}

describe_interventions <- function(iv) {
  # A named list (by what each acts on: "contacts", or a parameter) of interventions; a list of such lists, for scenarios, is several
  if (is.null(iv)) return(NULL)
  if (is_intervention(iv)) iv <- list(contacts = iv)
  if (length(iv) && all(vapply(iv, function(x) is.list(x) && !is_intervention(x), logical(1)))) iv <- iv[[1]] # Scenarios: the first
  out <- lapply(iv, describe_intervention)
  names(out) <- names(iv)
  out
}

describe_vaccination <- function(v) {
  if (is.null(v)) return(NULL)
  if (!is_vaccination(v)) v <- v[[1]] # Scenarios: the first
  if (is.null(v)) return(NULL)
  list(
    name = v$name,
    time_begin = unname(lapply(seq_len(NROW(v$time_begin)), function(i) as.numeric(as.matrix(v$time_begin)[i, ]))),
    time_end = unname(lapply(seq_len(NROW(v$time_end)), function(i) as.numeric(as.matrix(v$time_end)[i, ]))),
    nu = unname(lapply(seq_len(NROW(v$nu)), function(i) as.numeric(as.matrix(v$nu)[i, ])))
  )
}

recorder <- function(model) {
  defaults <- formals(get(model, envir = asNamespace("epidemics")))
  function(population, ...) {
    given <- list(...)
    call <- defaults
    call[names(given)] <- given
    call$population <- NULL
    pars <- list()
    for (key in names(call)) {
      val <- call[[key]]
      if (key %in% c("intervention", "vaccination", "time_dependence", "population_change")) next
      if (is.language(val)) val <- tryCatch(eval(val, envir = c(list(population = population), given)), error = function(e) NULL)
      if (is.numeric(val)) pars[[key]] <- as.numeric(val)
    }
    run <- list(
      model = model,
      population = describe_population(population),
      pars = pars,
      intervention = describe_interventions(given$intervention),
      vaccination = describe_vaccination(given$vaccination),
      time_dependence = names(given$time_dependence),
      population_change = !is.null(given$population_change),
      scenarios = (is.list(given$intervention) && length(given$intervention) > 1 && !is_intervention(given$intervention[[1]])) ||
        (is.list(given$vaccination) && !is_vaccination(given$vaccination) && length(given$vaccination) > 1)
    )
    fake$.runs[[length(fake$.runs) + 1]] <- run
    invisible(NULL)
  }
}
for (model in c("model_default", "model_vacamole", "model_diphtheria", "model_ebola")) assign(model, recorder(model), envir = fake)

exprs <- parse(infile)
for (e in exprs) {
  tryCatch(eval(e, envir = fake), error = function(err) message("Skipped: ", paste(deparse(e), collapse = " "), "\n  ", conditionMessage(err)))
}
if (!length(fake$.runs)) stop("The script calls none of model_default(), model_vacamole(), model_diphtheria(), or model_ebola()")
out <- list(format = "epidemics", version = as.character(packageVersion("epidemics")), file = infile, runs = fake$.runs)
writeLines(toJSON(out, auto_unbox = TRUE, null = "null", digits = NA, pretty = TRUE), outfile)
