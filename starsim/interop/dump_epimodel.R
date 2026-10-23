# Write an EpiModel model to JSON, for ss.interop.load() in Starsim
#
#     Rscript dump_epimodel.R model.R model.json    # an R script that calls netsim(), dcm(), or icm()
#     Rscript dump_epimodel.R sim.rds model.json    # a saved result of netsim(), or a saved netest()
#
# Needs only EpiModel and jsonlite. The script is run with netest(), netdx(), netsim(), dcm(), and icm()
# replaced by functions that record their arguments, so nothing is fitted or simulated; each call of
# netsim(), dcm(), or icm() is written as a run (several calls, e.g. one per scenario, are several runs).
# Statements that fail are skipped with a message, and the rest of the script still runs.
options(keep.source = TRUE)
suppressMessages(library(EpiModel))
suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript dump_epimodel.R <model.R or .rds> <out.json>")
infile <- normalizePath(args[1])
outfile <- file.path(normalizePath(dirname(args[2])), basename(args[2]))
fake <- new.env()
fake$here <- dirname(infile)
fake$.runs <- list()

# Helpers to describe each object
scalars <- function(x) {
  # The values of a list that are numbers, strings, or logicals (not functions or other objects)
  x <- unclass(x)
  keep <- vapply(x, function(v) is.atomic(v) && !is.null(v), logical(1))
  x[keep]
}

fun_source <- function(f) {
  src <- attr(f, "srcref")
  if (!is.null(src)) paste(as.character(src), collapse = "\n") else paste(deparse(f), collapse = "\n")
}

builtin <- function(f) environmentName(environment(f)) %in% c("EpiModel", "base", "stats")

fun_name <- function(f) {
  # The name of a function in the global environment, if it has one
  for (name in ls(globalenv())) {
    obj <- get(name, envir = globalenv())
    if (is.function(obj) && identical(obj, f)) return(name)
  }
  NA
}

describe_net <- function(est) {
  # The formation model, its targets, the dissolution model, and the nodes and their attributes
  nw <- if (!is.null(est$nw)) est$nw else est$newnetwork
  formation <- est$formation
  terms <- as.list(attr(stats::terms(formation), "variables"))[-1]
  terms <- lapply(terms, function(t) {
    call <- as.list(t)
    targs <- lapply(call[-1], function(a) if (is.language(a)) deparse(a) else a)
    list(term = as.character(call[[1]]), args = targs, text = deparse(t))
  })
  stats <- NULL # The name of each statistic, for the targets in order (a term can have several)
  try({
    f <- formation
    f <- stats::as.formula(paste("nw", paste(deparse(f), collapse = "")))
    environment(f) <- environment()
    s <- summary(f)
    stats <- names(s)
  }, silent = TRUE)
  attrs <- list()
  if (!is.null(nw)) {
    for (a in setdiff(network::list.vertex.attributes(nw), c("na", "vertex.names"))) {
      attrs[[a]] <- network::get.vertex.attribute(nw, a)
    }
  }
  diss <- est$coef.diss
  list(
    n = if (!is.null(nw)) network::network.size(nw) else NA,
    formation = paste(deparse(formation), collapse = ""),
    terms = terms,
    stats = stats,
    target_stats = est$target.stats,
    dissolution = paste(deparse(diss$dissolution), collapse = ""),
    duration = diss$duration,
    d_rate = if (is.null(diss$d.rate)) 0 else diss$d.rate,
    attributes = attrs
  )
}

describe_control <- function(control) {
  modules <- list()
  for (name in names(control)) {
    f <- control[[name]]
    if (endsWith(name, ".FUN") && is.function(f)) {
      if (builtin(f)) {
        modules[[name]] <- list(builtin = TRUE)
      } else {
        modules[[name]] <- list(builtin = FALSE, name = fun_name(f), source = fun_source(f))
      }
    }
  }
  list(values = scalars(control), modules = modules)
}

record <- function(kind, x, param, init, control) {
  nets <- list()
  if (kind == "netsim") {
    ests <- if (inherits(x, "netest")) list(x) else x
    for (i in seq_along(ests)) nets[[i]] <- describe_net(ests[[i]])
    names(nets) <- if (is.null(names(ests))) NULL else names(ests)
  }
  ctrl <- describe_control(control)
  run <- list(kind = kind, type = if (is.null(control$type)) NA else control$type, networks = nets,
              param = scalars(param), init = scalars(init), control = ctrl$values, modules = ctrl$modules)
  if (kind == "dcm" && is.function(control$new.mod)) {
    run$modules[["new.mod"]] <- list(builtin = FALSE, name = fun_name(control$new.mod), source = fun_source(control$new.mod))
  }
  .runs[[length(.runs) + 1]] <<- run
  invisible(structure(list(), class = "epimodel_dump"))
}

# Stand-ins for the functions that fit and simulate
fake$netest <- function(nw, formation, target.stats, coef.diss, ...) {
  structure(list(nw = nw, formation = formation, target.stats = target.stats, coef.diss = coef.diss), class = "netest")
}
fake$netdx <- function(...) invisible(NULL)
fake$netsim <- function(x, param, init, control) record("netsim", x, param, init, control)
fake$dcm <- function(param, init, control) record("dcm", NULL, param, init, control)
fake$icm <- function(param, init, control) record("icm", NULL, param, init, control)
fake$source <- function(file, ...) {
  # A script sourced by a path relative to somewhere else is looked for beside the model
  if (!file.exists(file) && file.exists(file.path(here, basename(file)))) file <- file.path(here, basename(file))
  base::source(file, ...)
}
fake$main <- function() {
  if (grepl("\\.rds$", infile, ignore.case = TRUE)) {
    x <- readRDS(infile)
    if (inherits(x, "netsim")) {
      nets <- lapply(x$nwparam, function(p) list(formation = p$formation, target.stats = p$target.stats, coef.diss = p$coef.diss))
      record("netsim", nets, x$param, x$init, x$control)
    } else if (inherits(x, "netest")) {
      record("netsim", x, list(), list(), list())
    } else if (inherits(x, "dcm") || inherits(x, "icm")) {
      record(class(x)[1], NULL, x$param, x$init, x$control)
    } else {
      stop("The file holds a ", class(x)[1], ", not a result of netsim(), netest(), dcm(), or icm()")
    }
  } else {
    owd <- setwd(here)
    on.exit(setwd(owd))
    for (expr in parse(infile, keep.source = TRUE)) {
      tryCatch(eval(expr, globalenv()), error = function(e) {
        message("Skipped (", conditionMessage(e), "): ", substr(paste(deparse(expr), collapse = " "), 1, 80))
      })
    }
  }
  if (!length(.runs)) stop("The script did not call netsim(), dcm(), or icm()")

  # Other functions the script defines, which the modules may call
  funcs <- list()
  used <- unlist(lapply(.runs, function(r) lapply(r$modules, function(m) m$name)))
  for (name in ls(globalenv())) {
    obj <- get(name, envir = globalenv())
    if (is.function(obj) && !builtin(obj) && !(name %in% used)) funcs[[name]] <- fun_source(obj)
  }
  out <- list(format = "epimodel", file = infile, version = as.character(packageVersion("EpiModel")), runs = .runs, functions = funcs)
  write_json(out, outfile, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE)
  message("Wrote ", length(.runs), " run(s) to ", outfile)
}

# Run with everything in its own environment, out of the way of rm(list = ls()) in the script
fake$infile <- infile
fake$outfile <- outfile
for (name in c("scalars", "fun_source", "builtin", "fun_name", "describe_net", "describe_control", "record")) assign(name, get(name), envir = fake)
for (name in ls(fake)) if (is.function(fake[[name]])) environment(fake[[name]]) <- fake
attach(fake, name = "epimodel_dump", warn.conflicts = FALSE)
rm(list = ls())
main()
