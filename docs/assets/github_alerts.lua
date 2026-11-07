-- Render GitHub alerts (> [!NOTE], > [!IMPORTANT], ...) as Quarto callouts, so that the README's alerts look the same on the docs home page
local types = {NOTE='note', TIP='tip', IMPORTANT='important', WARNING='warning', CAUTION='caution'}

function BlockQuote(el)
  local first = el.content[1]
  if not first or first.t ~= 'Para' or not first.content[1] or first.content[1].t ~= 'Str' then
    return nil
  end
  local kind = types[first.content[1].text:match('^%[!(%u+)%]$') or '']
  if not kind then
    return nil
  end
  first.content:remove(1) -- The marker
  if first.content[1] and (first.content[1].t == 'SoftBreak' or first.content[1].t == 'Space') then
    first.content:remove(1)
  end
  return quarto.Callout({type=kind, content=el.content})
end
