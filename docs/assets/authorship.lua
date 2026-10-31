-- Show a page's authorship as a badge at the top right, with the note on hover, e.g.:
--
-- authorship:
--   human: 90
--   ai: 10
--   note: Humans wrote everything, except some of the code examples were AI-generated

function Pandoc(doc)
  local a = doc.meta.authorship
  if not a or not quarto.doc.is_format('html') then return doc end

  local function get(key) return a[key] and pandoc.utils.stringify(a[key]) end
  local parts = {}
  if get('human') then table.insert(parts, '🧠' .. get('human') .. '%') end
  if get('ai') then table.insert(parts, '🤖' .. get('ai') .. '%') end
  local badge = table.concat(parts, ' ')
  local tip = 'Authorship: ' .. (get('note') or badge)

  local html = string.format('<div class="authorship" title="%s">%s</div>', tip:gsub('"', '&quot;'), badge)
  table.insert(doc.blocks, 1, pandoc.RawBlock('html', html))
  return doc
end
